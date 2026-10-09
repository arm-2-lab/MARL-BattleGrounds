"""Compare immutable tournament records through the public researcher interface.

Run ``python examples/canonical_tournament.py --help`` after installing MARL-BGs.
The fixture command creates clearly artificial local records, then exercises the
custom configuration route. It never installs an official release. The compare
command needs a separately released bundle and prepared assets. The read command
uses the host-only reader and performs no game, fitting or file repair.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.results import TournamentResult


def fixture(
    directory: Path,
    *,
    entrants: int,
    metrics: Literal["priority", "full", "none"],
    save: bool,
) -> None:
    """Create artificial records, read a complete field and optionally resume it.

    directory is explicit fixture storage. entrants selects 2 through 32 example
    versions; twelve is only this example's default. metrics is
    priority, full or none. save=False creates no result directory; the source
    fixture files still exist because this command explicitly creates them.
    True saves a result, reloads bounded tables and resumes without new games.
    No learned-method or official qualification claim is made.
    """
    if __package__:
        from .canonical_fixture import build_record_bundle
    else:
        from canonical_fixture import build_record_bundle

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.tournament_assets import (
        prepare_tournament_assets,
    )

    # Two spawn pairs per map give each matchup enough units for intervals.
    bundle = build_record_bundle(
        directory / "source",
        entrants=entrants,
        full=metrics == "full",
        pairs_per_map=2,
    )
    prepared = prepare_tournament_assets(
        bundle["config"],
        roles=("outcomes_priority", "full_report")
        if metrics == "full"
        else ("outcomes_priority",),
        download=False,
    )
    assert not prepared["missing"]
    result = marl_bgs.run_tournament(
        config=prepared["config"],
        metrics=metrics,
        output_dir=directory / "results" if save else None,
    )
    print("Artificial Fixture — Not A Released Snapshot")
    print("Imported Package:", marl_bgs.__file__)
    print("Status:", result.status)
    print("Rankings:", result.table("tournament_rankings"))
    if metrics != "none":
        print("Headlines:", result.table("tournament_headline_metrics"))
        show_entrants(result, f"{metrics}_metrics")
    for batch in result.iter_table("matches", rows=32):
        print("Original Games In This Batch:", len(batch["episode_id"]))
    if not save:
        assert result.run_dir is None
        return
    assert result.run_dir is not None
    print("Saved Run:", result.run_dir)
    loaded = marl_bgs.load_results(result.run_dir, phase="tournament")
    resumed = marl_bgs.run_tournament(resume_from=result.run_dir)
    assert loaded.status == resumed.status == "complete"
    print("Resumed Status:", resumed.status)


def show_entrants(result: "TournamentResult", table: str) -> None:
    """Print every entrant's own-side margin and kills, with Elo, one per line.

    result is any tournament result; table is "priority_metrics" or
    "full_metrics", whichever the run recorded, or "" to print nothing.
    system_metrics shows each entrant as Team A, so team_a_kills is always the
    entrant's own kills, whichever side it played. A challenger appears beside
    every member. The fixture's numbers are made up; this shows the call.
    """
    if not table:
        return
    field = result.system_metrics(
        table, columns=["score_difference", "team_a_kills", "team_a_deaths"]
    )
    print("Each Entrant As Team A:")
    for row in range(len(field["system_id"])):
        print(
            f"  {field['system_name'][row]}: Elo {field['elo'][row]:.0f}, "
            f"margin {field['score_difference'][row]:+.2f} per game "
            f"(95% {field['score_difference_ci_low'][row]:+.2f} to "
            f"{field['score_difference_ci_high'][row]:+.2f}), "
            f"kills {field['team_a_kills'][row]:.2f}, "
            f"deaths {field['team_a_deaths'][row]:.2f}; "
            f"Interval: {field['interval_status'][row]}"
        )


def compare(args: argparse.Namespace) -> None:
    """Run a selected released snapshot with explicit optional effects.

    args supplies the parsed snapshot/challenger, uniform budget, capture,
    save/resume and execution choices. Omitted values inherit saved conditions.
    Explicit prepare=true allows downloads first; otherwise execution is offline.
    A missing official snapshot or insufficient reused coverage raises the public
    actionable error. This command never admits or publishes its challenger.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.tournament_assets import (
        prepare_tournament_assets,
    )

    options: dict[str, Any] = {}
    if args.config is not None:
        options["config"] = args.config
    if args.system is not None:
        options["system"] = args.system
    for argument in ("games_per_opponent", "metrics", "save_replays", "rerun_existing"):
        value = getattr(args, argument)
        if value is not None:
            options[argument] = value
    if args.prepare:
        from marl_battlegrounds.evaluation.tournament_config import (
            resolve_tournament_config,
        )

        saved = (
            None
            if args.resume_from is None
            else marl_bgs.load_results(args.resume_from).metadata
        )
        resolved = resolve_tournament_config(
            args.config,
            official=True,
            saved=None if saved is None else saved["canonical_config"],
        )
        mode = (
            args.metrics
            if args.metrics is not None
            else "priority"
            if saved is None
            else saved["metrics"]
        )
        fresh = (
            args.rerun_existing
            if args.rerun_existing is not None
            else False
            if saved is None
            else saved["rerun_existing"]
        )
        challenger = args.system is not None or (
            saved is not None and saved.get("challenger_id") is not None
        )
        roles: list[str] = ["model"] if fresh or challenger else []
        if not fresh:
            roles.append("outcomes_priority")
            if mode == "full" or (
                saved is not None and saved.get("full_metrics_episodes")
            ):
                roles.append("full_report")
            if args.save_replays or (
                saved is not None and saved.get("replay_episodes")
            ):
                roles.append("replay")
        prepared = prepare_tournament_assets(
            resolved,
            cache_dir=args.cache_dir,
            roles=tuple(roles),
            download=True,
        )
        if prepared["missing"]:
            raise ValueError(f"Required assets remain missing: {prepared['missing']}")
        options["config"] = prepared["config"]
    result = marl_bgs.run_canonical_tournament(
        **options,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        num_envs=args.num_envs,
        chunk_size=args.chunk_size,
    )
    print("Snapshot:", result.snapshot_id)
    print(
        "Reused Games:", result.reused_games, "Executed Games:", result.executed_games
    )
    print("Rankings:", result.table("tournament_rankings"))
    recorded = result.metadata.get("metrics", "priority")
    show_entrants(result, f"{recorded}_metrics" if recorded != "none" else "")
    print("Replay Files:", result.replay_paths)
    if result.run_dir is not None:
        print("Saved Run:", result.run_dir)


def read(directory: Path, table: str) -> None:
    """Print bounded existing table batches without importing the simulator.

    directory is one exact saved run. table uses the shared public table names.
    Required external source files must remain available for dependent reads.
    No fitting, downloads, recovery or file writes occur.
    """
    import marl_battlegrounds as marl_bgs

    result = marl_bgs.load_results(directory)
    print("Imported Package:", marl_bgs.__file__)
    print("Status:", result.status)
    for batch in result.iter_table(table, rows=32):
        print(batch)


def run(
    *,
    command: str = "fixture",
    directory: str | Path | None = None,
    entrants: int = 12,
    metrics: Literal["priority", "full", "none"] | None = None,
    save: bool = False,
    config: str | Path | None = None,
    system: str | None = None,
    games_per_opponent: int | None = None,
    save_replays: int | None = None,
    rerun_existing: bool | None = None,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    num_envs: int = 128,
    chunk_size: int = 16,
    prepare: bool = False,
    cache_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    table: str = "matches",
) -> None:
    """Run the artificial fixture, a released comparison, or a saved table read.

    command is fixture (default), compare or read. Fixture uses directory,
    entrants (2..32), metrics (omitted means priority) and save. Saving requires
    an explicit directory; unsaved fixture files use a temporary folder.
    Compare forwards config/system, games_per_opponent, metrics, save_replays
    and rerun_existing only when non-null, preserving saved scientific settings.
    output_dir/resume_from, num_envs/chunk_size and cache_dir have the existing
    comparison meanings. prepare explicitly permits declared asset downloads.
    Read needs run_dir and prints table in bounded batches without numerical work.
    No mode publishes a policy or admits an official snapshot. API errors propagate.
    """
    for flag, value in (("save", save), ("prepare", prepare)):
        if type(value) is not bool:
            raise TypeError(f"{flag} must be true or false")
    if rerun_existing is not None and type(rerun_existing) is not bool:
        raise TypeError("rerun_existing must be true, false or None")
    if command == "read":
        if run_dir is None:
            raise ValueError("read needs run_dir naming saved results")
        read(Path(run_dir), table)
    elif command == "compare":
        compare(
            argparse.Namespace(
                config=config,
                system=system,
                games_per_opponent=games_per_opponent,
                metrics=metrics,
                save_replays=save_replays,
                rerun_existing=rerun_existing,
                output_dir=output_dir,
                resume_from=resume_from,
                num_envs=num_envs,
                chunk_size=chunk_size,
                prepare=prepare,
                cache_dir=cache_dir,
            )
        )
    elif command == "fixture":
        if save and directory is None:
            raise ValueError("fixture save=true needs directory for durable results")
        if directory is not None:
            fixture(
                Path(directory),
                entrants=entrants,
                metrics=metrics or "priority",
                save=save,
            )
        else:
            with TemporaryDirectory(prefix="marlbg-canonical-fixture-") as temporary:
                fixture(
                    Path(temporary),
                    entrants=entrants,
                    metrics=metrics or "priority",
                    save=False,
                )
    else:
        raise ValueError("command must be fixture, compare or read")


def main() -> int:
    """Compose settings and call the existing public tournament or reading owners."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("canonical_tournament", run)


if __name__ == "__main__":
    raise SystemExit(main())
