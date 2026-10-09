"""Play a language-model System against a field and in the endgame scenarios.

Start a model server first (docs/llm.md shows how), then play the
Season 0 field, the eight endgame scenarios, or both:

    JAX_PLATFORMS=cpu python -m marl_battlegrounds.experiments.llm_challenge \
        +field=season_0 models_dir=/path/hyperparameter-search output_dir=/path/field
    JAX_PLATFORMS=cpu python -m marl_battlegrounds.experiments.llm_challenge \
        'maps=[]' 'scenario_ids=[1,2,3,4,5,6,7,8]' output_dir=/path/scenarios

The stage builds one System with marl_bgs.llm.make_system and plays it as Team A.

Field games (``maps``) go through marl_bgs.evaluate, map by map: on each map the
System plays one pair of games (one from each spawn end) against every
opponent before the next map starts, so a time limit or a crash leaves every
opponent with the same number of maps played. Without +field the opponents are
the built-in TDM-Beta, TDM-Gamma and TDM-Alpha; names in ``skip`` (Random by
default) do not play.

Scenarios (``scenario_ids``) go through marl_bgs.evaluate_scenarios, one
scenario at a time, against each scenario's own scripted opponent, and give a
results table in the same format as the scenarios stage.

Results are saved after every pair of games and every scenario; running the
same command again skips what is already saved, and refuses to continue if a
different System (another prompt version, model or setting) made those saved
games. The stage trains nothing. Run
it with JAX_PLATFORMS=cpu so the games use the CPU and the model server keeps
the GPU.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from marl_battlegrounds.experiments import scenarios
# isort: on

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.policy_execution import System
    from marl_battlegrounds.evaluation.results import EvaluationResult

# Model costs saved for every pair of games and every scenario.
COSTS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "fallback_actions",
    "reply_failures",
)


def summarize_pair(result: EvaluationResult) -> dict[str, Any]:
    """Read one finished pair of games: outcomes, points, kills and model costs.

    Parameters
    ----------
    result : marl_battlegrounds.evaluation.results.EvaluationResult
        Result of ``marl_bgs.evaluate`` (or ``marl_bgs.load_results``) with the
        language-model System as Team A and priority or full metrics.

    Returns
    -------
    dict
        Totals over the games, from Team A's side: ``games``, ``wins``,
        ``draws``, ``losses``, ``points_for``, ``points_against``, ``kills``,
        ``deaths``, ``ticks`` (summed game lengths), and the model's
        ``model_calls``, ``input_tokens``, ``output_tokens``,
        ``fallback_actions`` and ``reply_failures`` in the completed games.

    Raises
    ------
    KeyError
        The result has no priority metrics or no language-model cost records.
    """
    import numpy as np

    from marl_battlegrounds import llm

    table = result.table("priority_metrics")

    def total(column: str) -> int:
        return int(np.sum(np.asarray(table[column])))

    costs = llm.call_summary(result)["completed_games"]["team_a"]
    return {
        "games": int(np.asarray(table["episode_length"]).size),
        "wins": total("team_a_win"),
        "draws": total("team_a_draw"),
        "losses": total("team_a_loss"),
        "points_for": total("team_a_score"),
        "points_against": total("team_b_score"),
        "kills": total("team_a_kills"),
        "deaths": total("team_a_deaths"),
        "ticks": total("episode_length"),
    } | {name: int(costs[name]) for name in COSTS}


def write_tables(
    pairs: Sequence[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    """Write ``results.csv`` and ``results.md``: one row per opponent.

    Parameters
    ----------
    pairs : sequence of dict
        Pair records from ``runs.json``: ``opponent``, ``seconds`` and the
        ``summarize_pair`` totals. Opponents keep their first-seen order.
    output_dir : pathlib.Path
        Folder for the two tables. Existing tables are replaced.

    Returns
    -------
    list of dict
        One row per opponent: ``opponent``, ``games``, ``wins``, ``draws``,
        ``losses``, per-game ``margin`` (points minus the opponent's points),
        ``kills`` and ``deaths``, mean ``ticks`` per game, the share of model
        calls whose reply failed and fell back to Stay (``fallback_percent``),
        and ``seconds_per_game`` (wall time of the opponent's pairs over its
        games; the two games of a pair play side by side).
    """
    rows: list[dict[str, Any]] = []
    for opponent in dict.fromkeys(pair["opponent"] for pair in pairs):
        mine = [pair for pair in pairs if pair["opponent"] == opponent]

        def total(key: str, mine: list[dict[str, Any]] = mine) -> int:
            return sum(int(pair[key]) for pair in mine)

        games = total("games")
        rows.append(
            {
                "opponent": opponent,
                "games": games,
                "wins": total("wins"),
                "draws": total("draws"),
                "losses": total("losses"),
                "margin": round(
                    (total("points_for") - total("points_against")) / games, 2
                ),
                "kills": round(total("kills") / games, 2),
                "deaths": round(total("deaths") / games, 2),
                "ticks": round(total("ticks") / games, 1),
                "fallback_percent": round(
                    100 * total("fallback_actions") / max(total("model_calls"), 1), 3
                ),
                "seconds_per_game": round(
                    sum(float(pair["seconds"]) for pair in mine) / games, 1
                ),
            }
        )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0]) if rows else ["opponent"]
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Opponent | Games | W · D · L | Margin | Kills | Deaths | Ticks "
        "| Fallback % | s / game |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {row['opponent']} | {row['games']} | {row['wins']} · {row['draws']} · "
        f"{row['losses']} | {row['margin']:+.2f} | {row['kills']:.2f} | "
        f"{row['deaths']:.2f} | {row['ticks']:.1f} | {row['fallback_percent']:.3f} | "
        f"{row['seconds_per_game']:.1f} |"
        for row in rows
    ]
    (output_dir / "results.md").write_text(
        "Team A is the language model. "
        "Margin, kills, deaths and ticks are per game.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return rows


def check_saved_system(system: System, run_dir: str) -> None:
    """Refuse to reuse saved games that a different System played as Team A.

    Parameters
    ----------
    system : System
        The language-model System this run plays as Team A, from
        ``llm.make_system``.
    run_dir : str
        A saved pair's or scenario's run folder, as listed in ``runs.json`` or
        ``scenario_runs.json``.

    Raises
    ------
    ValueError
        The folder's ``run_details.json`` is missing, unreadable or records no
        Team A System (for example after the output folder was moved, since the
        listed paths are absolute); or any recorded pass's Team A
        ``parameters_digest`` differs from this System's. The digest covers the
        System's whole declared identity, including ``llm.TEXT_VERSION``, the
        model, its revision and the generation settings, so games made with
        another prompt version or setting are never mixed into one table. The
        message names the folder and, for a difference, both digests. Use a
        fresh ``output_dir``.

    Notes
    -----
    Reads only that file. The evaluator writes Team A's System under
    ``passes[*]["system_ids"]["team_a"]`` and its description under
    ``systems``; every recorded pass must match.
    """
    path = Path(run_dir) / "run_details.json"
    current: object = (system.components or ({},))[0].get("parameters_digest")
    try:
        details = json.loads(path.read_text(encoding="utf-8"))
        saved: set[object] = {
            details["systems"][entry["system_ids"]["team_a"]]["components"][0][
                "parameters_digest"
            ]
            for entry in details["passes"].values()
        }
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        saved = set[object]()
    if not saved:
        raise ValueError(
            f"No readable record of the saved games' System in {run_dir}; "
            "use a fresh output_dir"
        )
    if saved != {current}:
        raise ValueError(
            f"The saved games in {run_dir} were made by a different System "
            f"(saved {sorted(map(str, saved))}, this one {current}): another "
            "prompt version, model or setting; use a fresh output_dir"
        )


def check_saved_runs(
    system: System, settings: Mapping[str, Any], output_dir: Path
) -> None:
    """Check every saved field pair and scenario before any opponent loads or game runs.

    Parameters
    ----------
    system : System
        The language-model System this run plays as Team A.
    settings : mapping
        The stage settings; only ``maps`` and ``scenario_ids`` are read, to
        check just the modes this run plays.
    output_dir : pathlib.Path
        The stage's output folder, holding ``runs.json`` and
        ``scenarios/scenario_runs.json`` from earlier runs, if any.

    Raises
    ------
    ValueError
        A saved list (``runs.json`` or ``scenario_runs.json``) cannot be read,
        or ``check_saved_system`` refuses a saved entry. Nothing is loaded or
        played first, so a wrong folder fails at once instead of after hours
        of field games.
    """
    lists: list[tuple[Path, str]] = []
    if settings["maps"]:
        lists.append((output_dir / "runs.json", "pairs"))
    if settings["scenario_ids"]:
        lists.append((output_dir / "scenarios" / "scenario_runs.json", "scenarios"))
    for path, key in lists:
        if not path.exists():
            continue
        try:
            run_dirs = [
                str(entry["run_dir"])
                for entry in json.loads(path.read_text(encoding="utf-8"))[key]
            ]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(
                f"Cannot read the list of saved games in {path}; use a fresh output_dir"
            ) from error
        for run_dir in run_dirs:
            check_saved_system(system, run_dir)


def play_field(
    system: System,
    opponents: Sequence[tuple[str, str]],
    settings: Mapping[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Play the System against every opponent, map by map, saving each pair.

    Parameters
    ----------
    system : System
        The language-model System, Team A in every game.
    opponents : sequence of (str, str)
        Display name and ``marl_bgs.load_method`` reference of each opponent,
        in playing order. Names must give distinct ``scenarios.folder_name``
        folders.
    settings : mapping
        The stage settings: ``maps``, ``games_per_map`` (a positive even
        number: pairs of games from both spawn ends, played side by side),
        ``seed``, ``max_steps``, ``metrics`` and ``time_limit_hours``.
    output_dir : pathlib.Path
        Existing folder. Each pair's games go under
        ``map-<id>/<opponent folder>/``; ``runs.json`` lists every pair.

    Returns
    -------
    list of dict
        Every saved pair, earlier runs' first: ``map``, ``opponent``,
        ``run_dir``, the ``summarize_pair`` totals and ``seconds``.

    Raises
    ------
    ValueError
        A pair in ``runs.json`` was not made by this System
        (``check_saved_system``).

    Notes
    -----
    Every opponent is loaded once, before the first game, so its compiled
    program is reused on every map. Pairs already in ``runs.json`` are
    skipped, after ``check_saved_system`` confirms, before any opponent loads,
    that this System made every one of them. No new pair starts once the
    elapsed time plus the longest pair so far would pass ``time_limit_hours``;
    a pair already playing finishes. The
    evaluator opens and closes the System's model client; the model server
    keeps running.
    """
    import marl_battlegrounds as marl_bgs

    games = int(settings["games_per_map"])
    runs_path = output_dir / "runs.json"
    pairs: list[dict[str, Any]] = (
        json.loads(runs_path.read_text(encoding="utf-8"))["pairs"]
        if runs_path.exists()
        else []
    )
    for pair in pairs:
        check_saved_system(system, pair["run_dir"])
    loaded = {name: marl_bgs.load_method(method) for name, method in opponents}
    done = {(pair["map"], pair["opponent"]) for pair in pairs}
    limit = 3600 * float(settings["time_limit_hours"])
    started = time.perf_counter()
    longest = 0.0
    for map_id in settings["maps"]:
        for name, _ in opponents:
            if (map_id, name) in done:
                continue
            if time.perf_counter() - started + longest > limit:
                print(f"Time limit: stopped before map {map_id}, {name}")
                return pairs
            pair_started = time.perf_counter()
            result = marl_bgs.evaluate(
                system,
                loaded[name],
                maps=[map_id],
                num_episodes=games,
                num_envs=games,
                spawn_mode="paired",
                seed=settings["seed"],
                max_steps=settings["max_steps"],
                metrics=settings["metrics"],
                save_replays=games,
                chunk_size=1,
                output_dir=output_dir / f"map-{map_id}" / scenarios.folder_name(name),
            )
            seconds = time.perf_counter() - pair_started
            longest = max(longest, seconds)
            pairs.append(
                {"map": map_id, "opponent": name, "run_dir": str(result.run_dir)}
                | summarize_pair(result)
                | {"seconds": round(seconds, 1)}
            )
            runs_path.write_text(
                json.dumps({"settings": dict(settings), "pairs": pairs}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            last = pairs[-1]
            print(
                f"map {map_id} vs {name}: {last['wins']}-{last['draws']}-"
                f"{last['losses']}, {last['ticks']} ticks in {seconds:.0f} s",
                flush=True,
            )
    return pairs


def play_scenarios(
    system: System, settings: Mapping[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    """Play the System as Team A in each endgame scenario, saving each one.

    Parameters
    ----------
    system : System
        The language-model System.
    settings : mapping
        The stage settings: ``model`` (the System's row name in the table),
        ``scenario_ids`` (played in this order), ``scenario_games`` (games per
        scenario), ``scenario_envs`` (games played side by side),
        ``scenario_seed`` and ``metrics``.
    output_dir : pathlib.Path
        Existing folder. Each scenario's games go under ``scenario-<id>/``;
        ``scenario_runs.json`` lists every scenario played, ``results.csv``
        and ``results.md`` give wins, draws and losses in the scenarios
        stage's format, and ``costs.csv`` the model costs per scenario.

    Returns
    -------
    list of dict
        One record per saved scenario, earlier runs' first: ``scenario``,
        ``run_dir``, ``games``, ``wins``, ``draws``, ``losses``, the model
        costs of the completed games and ``seconds``.

    Raises
    ------
    ValueError
        A scenario in ``scenario_runs.json`` was not made by this System
        (``check_saved_system``), or ``evaluate_scenarios`` saved no folder.

    Notes
    -----
    Each scenario starts from its authored position and meets its own
    scripted opponent, TDM-Alpha or TDM-Beta. Scenarios already in
    ``scenario_runs.json`` are skipped, after ``check_saved_system`` confirms,
    before any game starts, that this System made every one of them.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import llm

    runs_path = output_dir / "scenario_runs.json"
    records: list[dict[str, Any]] = (
        json.loads(runs_path.read_text(encoding="utf-8"))["scenarios"]
        if runs_path.exists()
        else []
    )
    for record in records:
        check_saved_system(system, record["run_dir"])
    done = {record["scenario"] for record in records}
    for scenario in settings["scenario_ids"]:
        if scenario in done:
            continue
        started = time.perf_counter()
        (result,) = marl_bgs.evaluate_scenarios(
            system,
            scenario_ids=[scenario],
            num_episodes=settings["scenario_games"],
            seed=settings["scenario_seed"],
            num_envs=settings["scenario_envs"],
            metrics=settings["metrics"],
            output_dir=output_dir,
        )
        if result.run_dir is None:
            raise ValueError("evaluate_scenarios saved no run folder")
        costs = llm.call_summary(result)["completed_games"]["team_a"]
        records.append(
            {"scenario": scenario, "run_dir": str(result.run_dir)}
            | scenarios.count_outcomes(result.run_dir)
            | {name: int(costs[name]) for name in COSTS}
            | {"seconds": round(time.perf_counter() - started, 1)}
        )
        runs_path.write_text(
            json.dumps({"scenarios": records}, indent=2) + "\n", encoding="utf-8"
        )
        last = records[-1]
        print(
            f"scenario {scenario}: {last['wins']}-{last['draws']}-{last['losses']}"
            f" in {last['seconds']:.0f} s",
            flush=True,
        )
    scenarios.write_tables(
        [
            {
                "system": settings["model"],
                "scenario_runs": {
                    str(record["scenario"]): record["run_dir"] for record in records
                },
            }
        ],
        [record["scenario"] for record in records],
        output_dir,
    )
    fields = ["scenario", "games", *COSTS, "seconds"]
    with (output_dir / "costs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return records


@hydra.main(version_base="1.3", config_path="configs", config_name="llm_challenge")
def main(config: DictConfig) -> dict[str, list[dict[str, Any]]]:
    """Build the language-model System, then play the field and the scenarios.

    Parameters
    ----------
    config : omegaconf.DictConfig
        The System's ``model``, ``server_url``, ``model_revision``,
        ``history_turns``, ``failure_policy`` and ``records`` (passed to
        ``marl_bgs.llm.make_system``); for the field, ``systems`` (a list of
        ``name`` and ``method``; ``models_dir`` fills ``${models_dir}`` in
        method paths), ``skip`` and the ``play_field`` settings; for the
        scenarios, the ``play_scenarios`` settings; ``metrics``; and the
        required ``output_dir``. An empty ``maps`` or ``scenario_ids`` skips
        that mode. Relative paths use the folder the command was started from.

    Returns
    -------
    dict
        ``field``: the ``write_tables`` rows, one per opponent (empty without
        maps); ``scenarios``: the ``play_scenarios`` records.

    Raises
    ------
    ValueError
        Both modes are empty; or maps are given and no opponent is left after
        ``skip``, two opponents share a folder name, or ``games_per_map`` is
        not a positive even number; or games already saved in ``output_dir``
        were made by a different System (``check_saved_system``).

    Notes
    -----
    Field results go in ``output_dir`` (``runs.json``, ``results.csv``,
    ``results.md``) and scenario results in ``output_dir/scenarios``.
    """
    from marl_battlegrounds import llm

    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    if not settings["maps"] and not settings["scenario_ids"]:
        raise ValueError("give maps, scenario_ids or both")
    opponents = [
        (str(entry["name"]), str(entry["method"]))
        for entry in settings["systems"]
        if entry["name"] not in settings["skip"]
    ]
    if settings["maps"]:
        if not opponents:
            raise ValueError("no opponent is left after skip")
        folders = {scenarios.folder_name(name) for name, _ in opponents}
        if len(folders) != len(opponents):
            raise ValueError("opponent names must give distinct folder names")
        games = int(settings["games_per_map"])
        if games <= 0 or games % 2:
            raise ValueError("games_per_map must be a positive even number")
    output_dir = Path(to_absolute_path(str(Path(settings["output_dir"]).expanduser())))
    output_dir.mkdir(parents=True, exist_ok=True)
    system = llm.make_system(
        settings["model"],
        settings["server_url"],
        model_revision=settings["model_revision"],
        history_turns=settings["history_turns"],
        failure_policy=settings["failure_policy"],
        records=settings["records"],
    )
    check_saved_runs(system, settings, output_dir)
    rows: list[dict[str, Any]] = []
    if settings["maps"]:
        rows = write_tables(
            play_field(system, opponents, settings, output_dir), output_dir
        )
        print(output_dir / "results.md")
    played: list[dict[str, Any]] = []
    if settings["scenario_ids"]:
        (output_dir / "scenarios").mkdir(exist_ok=True)
        played = play_scenarios(system, settings, output_dir / "scenarios")
        print(output_dir / "scenarios" / "results.md")
    return {"field": rows, "scenarios": played}


if __name__ == "__main__":
    prepare_hydra()
    main()
