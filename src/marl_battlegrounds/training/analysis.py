"""Summarize saved training and frozen validation without running a learner.

The runner, CLI and researchers share selection, paired uncertainty and report
generation here. Evaluation remains the owner of recorded game scores and metrics.
Score fields are points and kill fields are kills: under the Red Zone rule a
death can give 2 points but is still one kill, so _kill_columns names where each
result's kills live. Reports read committed trainer records and never rewrite
original evidence. Training and validation Red Zone depths keep separate labels;
missing historical values stay unknown, distinct from recorded zero.
analyze_screen also reports an unfinished configuration screen, retaining shared
Random checks as one statistical task while plotting each case's own timing.
list_runs(progress=True) adds each run's progress, validation checks and
per-game training results, read from saved files for the study watch view.
A grouped analysis also writes recipe learning curves and saved recipe picks,
where a declared base keeps its place unless a challenger beats it clearly.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import numpy as np

from marl_battlegrounds.training._launch import read_declaration
from marl_battlegrounds.training.config import (
    _integer as _integer,  # pyright: ignore[reportPrivateUsage]
)

Record = dict[str, Any]
# Row columns that hold each team's recorded kills (priority metrics) and each
# team's points (episode scores). With a positive Red Zone depth a Red Zone
# death gives 2 points but is still one kill, so the two can differ.
_KILL_COLUMNS = ("team_a_kills", "team_b_kills")
_SCORE_COLUMNS = ("team_a_score", "team_b_score")


def list_runs(
    root: str | Path, *, progress: bool = False
) -> tuple[dict[str, object], ...]:
    """List saved training runs and declared study cases without loading models.

    Parameters
    ----------
    root : str or Path
        A run, study, or parent folder. Folders are walked in sorted order,
        stopping inside each training run.
    progress : bool, default=False
        False keeps the ordinary rows exactly as before. True adds one
        ``progress`` field to every readable training run row (see
        _run_progress): speed, time left, every finished routine check, the
        training windows and the training results by opponent group. Rows of
        cases that never started, unreadable runs and evaluation folders get
        no ``progress`` field. This also reads validation_results.json and
        training_updates.jsonl, so it costs more than the ordinary listing.

    Returns
    -------
    tuple of dict
        Rows sorted by run_dir, with run_dir, run_id, method, seed, status,
        env_steps, completed_updates and saved actor paths. Malformed run
        records get status='unreadable' and a reason. Study cases that never
        started keep their declared status and path. An open recovery marker
        takes precedence over stale complete status.

    Raises
    ------
    OSError
        root is missing or is not a folder.

    Notes
    -----
    Read-only and JAX-free: no file is written and no worker or model is
    opened. Only small JSON records are read unless progress is True.
    """
    from marl_battlegrounds.training.study import (
        _jobs,  # pyright: ignore[reportPrivateUsage]
    )

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    rows: dict[str, dict[str, object]] = {}
    for directory, children, filenames in os.walk(base, followlinks=False):
        children.sort()
        path = Path(directory)
        names = set(filenames)
        if "declaration.json" in names:
            try:
                declaration = _read_json(path / "declaration.json", {})
                state = _read_json(path / "study.json", {})
                if not isinstance(declaration, dict) or not isinstance(state, dict):
                    raise ValueError("Study declaration and status must be objects")
                declaration, state = cast(Record, declaration), cast(Record, state)
                for job in _jobs(declaration):
                    run = path / "cases" / job["id"] / "run"
                    case = state.get("cases", {}).get(job["id"], {})
                    config = job.get("config", {})
                    rows[str(run)] = {
                        "run_dir": str(run),
                        "run_id": None,
                        "method": config.get(
                            "method", job.get("parent_identity", {}).get("method")
                        ),
                        "seed": config.get("seed"),
                        "branch_seed": job.get("changes", {}).get("seed"),
                        "case_id": job["id"],
                        "status": case.get("status", "pending"),
                        "env_steps": None,
                        "completed_updates": None,
                        "reason": None if run.exists() else "Run has not started",
                    }
            except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
                rows[str(path)] = {
                    "run_dir": str(path),
                    "status": "unreadable",
                    "reason": str(error),
                }
        if not (
            {"latest_checkpoint.json", "training_updates.jsonl"} & names
            or "run_details.json" in names
        ):
            continue
        row: dict[str, object] = {"run_dir": str(path)}
        try:
            info = _read_json(path / "run_details.json", None)
            if not isinstance(info, dict):
                raise ValueError("Missing or malformed run_details.json")
            info = cast(Record, info)
            if "config" not in info:
                # Evaluation folders have their own run_details.json contract.
                continue
            config = info["config"]
            if not isinstance(config, dict) or not isinstance(info.get("run_id"), str):
                raise ValueError("Training run identity/config is invalid")
            config = cast(Record, config)
            status = _read_json(path / "status.json", {})
            if not isinstance(status, dict):
                raise ValueError("Saved status must be an object")
            status = cast(Record, status)
            if status and status.get("run_id") != info["run_id"]:
                raise ValueError("Run and status identities differ")
            row.update(
                run_id=info["run_id"],
                method=config.get("method", "mappo"),
                seed=config.get("seed"),
                status="recovering"
                if (path / "checkpoint_recovery.json").exists()
                else status.get("status", "incomplete"),
                env_steps=status.get("env_steps"),
                completed_updates=status.get("completed_updates"),
                final_actor=status.get("final_actor"),
                selected_actor=status.get("selected_actor"),
                reason=None,
            )
            if progress:
                row["progress"] = _run_progress(path, config, status)
        except (ValueError, OSError, KeyError, TypeError) as error:
            row.update(status="unreadable", reason=str(error))
        rows[str(path)] = {**rows.get(str(path), {}), **row}
        children.clear()
    return tuple(rows[key] for key in sorted(rows))


# Prices of the built-in combat/support reward parts. The owner is
# training/reward_recipes.py (_KILL, _RED_ZONE_KILL, _DAMAGE, _PRIEST_HEAL,
# _HEAL_RANGE, _DEATH, _RED_ZONE_DEATH, _ZONE_TICK). They are repeated here so
# the progress view can turn saved amounts back into counts without importing
# JAX. Keep both copies equal when a price changes.
_RECIPE_PRICES: dict[str, float] = {
    "enemy_death_credit": 0.05,
    "enemy_red_zone_death_credit": 0.10,
    "enemy_damage_hp": 0.001,
    "priest_effective_healing_hp": 0.001,
    "priest_heal_range_pairs": 0.001,
    "own_team_deaths": -0.05,
    "own_red_zone_deaths": -0.10,
    "own_zone_ticks": -0.002,
}
# Parts every learner receives in equal amounts (reward_recipes._SHARED_PARTS).
_RECIPE_SHARED_PARTS = frozenset(
    ("own_team_deaths", "own_red_zone_deaths", "priest_heal_range_pairs")
)
_RECIPE_FORMS = {
    "marl_battlegrounds.training.reward_recipes:combat_support_reward": "agent",
    "marl_battlegrounds.training.reward_recipes:combat_support_team_reward": "team",
}
# One progress window is this many consecutive training_updates.jsonl rows.
# Each row is one collection round (about 25 PPO learner updates per window).
PROGRESS_WINDOW_ROWS = 25
# Parsed update logs, keyed by path, reused while size and change time match.
_UPDATE_CACHE: dict[str, tuple[tuple[int, int], Record]] = {}


def _run_progress(path: Path, config: Mapping[str, Any], status: Record) -> Record:
    """Collect one run's live progress from its saved files, for the study view.

    Parameters
    ----------
    path : Path
        A training run folder (it holds run_details.json).
    config : mapping
        The run's saved settings from run_details.json. total_env_steps and
        reward are read.
    status : dict
        The run's saved status.json, or an empty dict before the first save.

    Returns
    -------
    dict
        ``total_env_steps`` (int or None), ``phase`` (str or None),
        ``steps_per_second`` (recent training transitions per second, falling
        back to the whole-run average; None when unknown),
        ``elapsed_seconds`` (run time without gaps between attempts, or None),
        ``seconds_left`` (remaining transitions divided by that speed; it
        leaves out later validation and saving; None when unknown),
        ``checks`` (see _progress_checks), and ``windows``,
        ``opponent_groups``, ``reward_form`` and ``update_rows`` (see
        _update_progress).

    Raises
    ------
    ValueError
        validation_results.json is malformed JSON or not a list.
    OSError
        A file exists but cannot be read.

    Notes
    -----
    Read-only and JAX-free. Missing files give empty lists and None values.
    """
    total = status.get("total_env_steps", config.get("total_env_steps"))
    steps = status.get("env_steps")
    speed = status.get("recent_transitions_per_second") or status.get(
        "transitions_per_second"
    )
    left = None
    if (
        isinstance(total, int)
        and isinstance(steps, int)
        and isinstance(speed, (int, float))
        and speed > 0
    ):
        left = max(0, total - steps) / float(speed)
    return {
        "total_env_steps": total,
        "phase": status.get("phase"),
        "steps_per_second": speed,
        "elapsed_seconds": status.get("elapsed_seconds"),
        "seconds_left": left,
        "checks": _progress_checks(path / "validation_results.json"),
        **_update_progress(
            path / "training_updates.jsonl",
            _RECIPE_FORMS.get(str(config.get("reward"))),
        ),
    }


def _mean_cell(cells: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """Average one cell field with equal weight per cell; None if any is missing."""
    values = [cell.get(key) for cell in cells]
    if not values or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        return None
    return math.fsum(cast(list[float], values)) / len(values)


def _progress_checks(path: Path) -> list[Record]:
    """Summarize every finished routine validation check, oldest first.

    Parameters
    ----------
    path : Path
        A run's validation_results.json. A missing file gives an empty list.

    Returns
    -------
    list of dict
        One record per complete check with purpose "routine", sorted by
        env_steps: ``env_steps``, ``checkpoint_id``, ``opponents`` (one dict
        per opponent in saved order with name, games, wins, draws, losses,
        score, the average game score: win 1, draw 0.5, loss 0, and that
        opponent's own ``points_for``, ``points_against``, ``margin``,
        ``kills_for`` and ``kills_against`` per game, each averaging its map
        cells with equal weight; None when a cell lacks the value),
        ``points_for`` and ``points_against`` (our and their points per game),
        ``margin`` (points for minus points against from _mean_point_margin,
        equal weight per map and opponent), ``kills_for`` and
        ``kills_against`` (kills per game), ``red_zone_kills`` (our points
        minus our kills: each Red Zone kill gives one extra point) and
        ``home_deaths`` (their points minus their kills: our deaths inside
        our own Red Zone). Every per-game value averages map/opponent cells
        with equal weight. A value the saved cells cannot give is None.

    Raises
    ------
    ValueError
        The file is malformed JSON or does not hold a list.
    """
    results = _read_json(path, [])
    if not isinstance(results, list):
        raise ValueError(f"Expected a list in {path.name}")
    checks: list[Record] = []
    for result in cast(list[object], results):
        if not isinstance(result, dict):
            continue
        result = cast(Record, result)
        if result.get("purpose") != "routine" or not result.get("complete"):
            continue
        cells = [
            cast(Record, cell)
            for cell in cast(list[object], result.get("cells") or [])
            if isinstance(cell, dict)
        ]
        names: list[str] = []
        for cell in cells:
            if cell.get("opponent") not in names:
                names.append(str(cell.get("opponent")))
        opponents: list[Record] = []
        for name in names:
            own = [cell for cell in cells if str(cell.get("opponent")) == name]
            counts = {
                key: sum(int(cell.get(key) or 0) for cell in own)
                for key in ("games", "wins", "draws", "losses")
            }
            own_for = _mean_cell(own, "mean_team_a_score")
            own_against = _mean_cell(own, "mean_team_b_score")
            opponents.append(
                {
                    "name": name,
                    **counts,
                    "score": (counts["wins"] + 0.5 * counts["draws"]) / counts["games"]
                    if counts["games"]
                    else None,
                    "points_for": own_for,
                    "points_against": own_against,
                    "margin": None
                    if own_for is None or own_against is None
                    else own_for - own_against,
                    "kills_for": _mean_cell(own, "mean_team_a_kills"),
                    "kills_against": _mean_cell(own, "mean_team_b_kills"),
                }
            )
        try:
            margin: float | None = _mean_point_margin(result)
        except ValueError:
            margin = None
        points_for = _mean_cell(cells, "mean_team_a_score")
        points_against = _mean_cell(cells, "mean_team_b_score")
        kills_for = _mean_cell(cells, "mean_team_a_kills")
        kills_against = _mean_cell(cells, "mean_team_b_kills")
        checks.append(
            {
                "env_steps": result.get("env_steps"),
                "checkpoint_id": result.get("checkpoint_id"),
                "opponents": opponents,
                "points_for": points_for,
                "points_against": points_against,
                "margin": margin,
                "kills_for": kills_for,
                "kills_against": kills_against,
                "red_zone_kills": None
                if points_for is None or kills_for is None
                else points_for - kills_for,
                "home_deaths": None
                if points_against is None or kills_against is None
                else points_against - kills_against,
            }
        )
    return sorted(checks, key=lambda check: int(check["env_steps"] or 0))


def _update_progress(path: Path, form: str | None) -> Record:
    """Read training_updates.jsonl into training windows and opponent groups.

    Parameters
    ----------
    path : Path
        A run's training_updates.jsonl. A missing file gives no windows and
        no groups. A partial last line (a row still being written) is left
        out. Rows are read as written; a later recovery may replace rows
        written after the latest checkpoint.
    form : {"agent", "team"} or None
        How the run's reward pays its parts. "agent" is
        reward_recipes.combat_support_reward (each agent gets its own events),
        "team" is combat_support_team_reward (every agent gets the team
        total). None means another or no reward recipe: saved amounts cannot
        be turned into counts, so every count is None.

    Returns
    -------
    dict
        ``update_rows`` (int), ``reward_form`` (form), ``windows`` and
        ``opponent_groups``. Each window covers PROGRESS_WINDOW_ROWS
        consecutive rows from the start (the last may be shorter) and holds
        ``rows``, ``env_steps`` (the last row's step count), ``games`` (games
        that ended in the window), ``length_sum`` (their ticks, None when a
        length is unavailable), ``points_for`` and ``points_against`` (sums
        over those games), ``counts`` (per part name: the window's total
        count, such as kill credits, enemy damage HP or own deaths; None when
        not exact), ``shaping`` (combat/support reward one learner agent got
        in the window, averaged over learner agents; None without saved
        parts) and ``agent_transitions`` (learner agent ticks). Each
        opponent group holds ``group`` ("current", "past", "pinned", "bots"
        or a named member's name), ``games``, ``wins``, ``draws``,
        ``losses``, ``length_sum`` (None if unknown) and ``start_env_steps``
        (the step its counts start at; above zero for a continued run), using
        each member's latest saved whole-run totals.

    Notes
    -----
    Counts divide each saved amount by its recipe price. Under "agent", the
    three shared parts use the per-agent mean (every learner gets the same
    amount) and the other parts the sum over learner agents. Under "team",
    every part uses the per-agent mean, which is the team amount. Counts are
    exact while every learner slot has a transition; under "team" they
    include slots a fixed partner controls. Parsed files are cached in this
    process by size and change time, so a refresh rereads only files that
    changed; every call returns a fresh copy the caller may change.
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {
            "update_rows": 0,
            "reward_form": form,
            "windows": [],
            "opponent_groups": [],
        }
    key = (stat.st_size, stat.st_mtime_ns)
    cached = _UPDATE_CACHE.get(str(path))
    if cached is not None and cached[0] == key and cached[1]["reward_form"] == form:
        return copy.deepcopy(cached[1])
    text = path.read_text()
    lines = text.split("\n")[:-1]
    rows: list[Record] = []
    for line in lines:
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(cast(Record, value))
    from marl_battlegrounds.policies import BUILTIN_POLICY_NAMES

    windows: list[Record] = []
    latest: dict[tuple[str, int], Record] = {}
    for start in range(0, len(rows), PROGRESS_WINDOW_ROWS):
        chunk = rows[start : start + PROGRESS_WINDOW_ROWS]
        games = 0
        length: int | None = 0
        points = [0.0, 0.0]
        recipient: dict[str, float] = {}
        mean: dict[str, float] = {}
        agents = 0
        parts_seen = False
        for row in chunk:
            for member in cast(list[Record], row.get("opponent_results") or []):
                latest[(str(member.get("kind")), int(member.get("snapshot", 0)))] = (
                    member
                )
                last = cast(Record, member.get("last_update") or {})
                if last.get("recorded") is False:
                    continue
                games += int(last.get("games") or 0)
                points[0] += float(last.get("points_for") or 0)
                points[1] += float(last.get("points_against") or 0)
                if length is not None and last.get("games"):
                    value = last.get("length_sum")
                    length = None if value is None else length + int(value)
            record = cast(Record | None, row.get("collected_reward_components"))
            if record is None:
                continue
            parts_seen = True
            agents += int(record.get("learner_agent_transitions") or 0)
            for name, amounts in cast(
                dict[str, dict[str, float]], record.get("components") or {}
            ).items():
                recipient[name] = recipient.get(name, 0.0) + amounts["recipient_sum"]
                mean[name] = mean.get(name, 0.0) + amounts["team_mean_sum"]
        counts: dict[str, float | None] = {}
        for name, price in _RECIPE_PRICES.items():
            if form is None or name not in mean:
                counts[name] = None
            elif form == "team" or name in _RECIPE_SHARED_PARTS:
                counts[name] = mean[name] / price
            else:
                counts[name] = recipient[name] / price
        windows.append(
            {
                "rows": len(chunk),
                "env_steps": chunk[-1].get("env_steps"),
                "games": games,
                "length_sum": length,
                "points_for": points[0],
                "points_against": points[1],
                "counts": counts,
                "shaping": math.fsum(mean.values()) if parts_seen else None,
                "agent_transitions": agents,
            }
        )
    groups: dict[str, Record] = {}
    for (kind, _snapshot), member in latest.items():
        name = str(member.get("name"))
        group = {"self": "current", "past": "past", "pin": "pinned"}.get(
            kind, "bots" if name in BUILTIN_POLICY_NAMES else name
        )
        total = cast(Record, member.get("cumulative") or {})
        entry = groups.setdefault(
            group,
            {
                "group": group,
                "games": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "length_sum": 0,
                "start_env_steps": 0,
            },
        )
        for field in ("games", "wins", "draws", "losses"):
            entry[field] += int(total.get(field) or 0)
        if entry["length_sum"] is not None:
            value = total.get("length_sum")
            entry["length_sum"] = (
                None
                if value is None and total.get("games")
                else entry["length_sum"] + int(value or 0)
            )
        entry["start_env_steps"] = max(
            entry["start_env_steps"], int(total.get("start_env_steps") or 0)
        )
    result: Record = {
        "update_rows": len(rows),
        "reward_form": form,
        "windows": windows,
        "opponent_groups": list(groups.values()),
    }
    _UPDATE_CACHE[str(path)] = (key, result)
    return copy.deepcopy(result)


def _kill_columns(record: Mapping[str, Any]) -> tuple[str, str]:
    """Name the row columns that hold Team A and Team B kills for one result.

    Parameters
    ----------
    record : mapping
        One validation or Random result: its task description plus summary,
        as validate_checkpoint, validate_random or a runner record returns it.

    Returns
    -------
    tuple of str
        ("team_a_kills", "team_b_kills") when the record carries a
        red_zone_depth (its task was built with the Red Zone rule, and its
        rows and cells hold recorded kills); otherwise ("team_a_score",
        "team_b_score"), because older records scored one point per death, so
        their scores were their kills. Cell means use the same names with a
        "mean_" prefix, for example "mean_team_a_kills".

    Notes
    -----
    The one owner of "which fields hold kills" for training reports
    (runner._random_progress, _screen_statistics and analyze_screen). Reads
    only the mapping's keys; no file or device work.
    """
    return _KILL_COLUMNS if "red_zone_depth" in record else _SCORE_COLUMNS


def _score(value: object) -> float:
    """Read one authoritative W/D/L score; reject missing or invented values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("A completed game needs a numerical system_game_score")
    result = float(value)
    if result not in (0.0, 0.5, 1.0):
        raise ValueError("system_game_score must be 0, 0.5 or 1")
    return result


def _bootstrap(
    blocks: Sequence[np.ndarray[Any, np.dtype[np.float64]]],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Resample whole independent blocks within maps, then weight maps equally.

    Each one-dimensional float64 array contains one map's block means. The caller
    must keep dependent games together inside each block. draws is positive and
    seed is a nonnegative integer. Return the 2.5/97.5 percentile score bounds.
    This host calculation uses its own NumPy generator and changes no input.
    """
    _integer(draws, "bootstrap draws", minimum=1)
    _integer(seed, "bootstrap seed")
    random = np.random.default_rng(seed)
    means = np.zeros(draws, np.float64)
    for values in blocks:
        indices = random.integers(len(values), size=(draws, len(values)))
        means += values[indices].mean(axis=1) / len(blocks)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def summarize_validation(
    rows: Sequence[Mapping[str, Any]],
    *,
    maps: Sequence[int],
    opponents: Sequence[str],
    seed_pairs: int,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 19_044_001,
    independent_opponents: bool = False,
    actual_kills: bool = False,
    sampling_evidence: Mapping[str, Any] | None = None,
) -> Record:
    """Reduce complete frozen-panel games with their shared random seeds intact.

    Parameters
    ----------
    rows : sequence of mappings
        Saved evaluation rows augmented with opponent and spawn_locations. Required
        fields are map_id, seed_id, opponent, spawn_locations and system_game_score.
        Each map/seed/opponent has exactly two ends. Optional scores (points), kills and
        lengths are reported only when every game in that cell supplies them.
    maps, opponents : sequences
        Nonempty distinct map IDs and frozen opponent labels in declared order.
    seed_pairs : int
        Positive independent paired seeds per map/opponent. Opponents share those
        seeds; different maps must use distinct seed IDs.
    bootstrap_draws, bootstrap_seed : int
        Positive replicate count and independent NumPy seed, default 2000/19044001.

    independent_opponents : bool, default=False
        True uses new-panel independent opponent roots. Spawn ends remain paired;
        every game must supply finite kills for mean kill difference (see
        actual_kills). False keeps the historical shared-seed uncertainty and
        result fields unchanged.
    actual_kills : bool, default=False
        True means the rows also carry recorded kills (team_a_kills and
        team_b_kills from evaluation's priority metrics). Each cell then also reports
        mean_team_a_kills and mean_team_b_kills, and mean_kill_difference uses
        these kills. Use it for every result built with a Red Zone depth,
        because a Red Zone death gives 2 points but is still one kill. False
        keeps the historical rule and bytes: kills are read from the Team A
        and Team B scores, which were equal to kills before the rule.
    sampling_evidence : mapping or None, default=None
        Shared versioned sampling evidence for these completed games. Unsupported
        sampling leaves ci_low/ci_high unavailable; conditional_ci_low/high retain
        only the mathematical calculation under an explicitly unsupported sampling
        assumption. Each stratum also needs at least two seed pairs. None preserves
        historical outputs and their original conditional interpretation. Supplied
        evidence keeps independent_blocks as the declared block count and adds
        supported_independent_sampling_units: zero for deterministic conditions
        and None when the sampling mechanism is unknown.

    Returns
    -------
    dict
        Equal-weight score, descriptive paired interval, W/D/L cells and complete
        counts. The interval concerns these fixed actors, not training-seed variation.
        Score fields are always points; kill fields are kills.

    Raises
    ------
    ValueError
        Coverage, score values, dimensions or shared seed pairing are incompatible.

    Notes
    -----
        Host-only. All opponents and both ends stay together in a bootstrap block.
        No original metrics are recomputed and no file or global RNG is changed.
    """
    if independent_opponents:
        summary = _independent_validation_summary(
            rows,
            maps=maps,
            opponents=opponents,
            seed_pairs=seed_pairs,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed,
            actual_kills=actual_kills,
        )
        return _validation_sampling_interval(summary, sampling_evidence, seed_pairs)
    _integer(seed_pairs, "seed_pairs", minimum=1)
    map_ids, names = tuple(maps), tuple(opponents)
    if not map_ids or not names or len(set(map_ids)) != len(map_ids):
        raise ValueError("Validation needs nonempty distinct maps and opponents")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Opponent labels must be nonempty and distinct")
    expected_count = len(map_ids) * len(names) * seed_pairs * 2
    if len(rows) != expected_count:
        raise ValueError("Validation has missing or extra games")
    indexed: dict[tuple[int, int, str, int], Mapping[str, Any]] = {}
    seeds: dict[int, set[int]] = {map_id: set() for map_id in map_ids}
    for row in rows:
        map_id = _integer(row.get("map_id"), "map_id")
        seed_id = _integer(row.get("seed_id"), "seed_id")
        end = _integer(row.get("spawn_locations"), "spawn_locations")
        name = row.get("opponent")
        if map_id not in seeds or name not in names or end not in (0, 1):
            raise ValueError("Validation game is outside its declared conditions")
        key = (map_id, seed_id, cast(str, name), end)
        if key in indexed:
            raise ValueError("Validation repeats a map/seed/opponent/spawn game")
        _score(row.get("system_game_score"))
        indexed[key] = row
        seeds[map_id].add(seed_id)
    seen_seeds: set[int] = set()
    blocks: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    cells: list[Record] = []
    for map_id in map_ids:
        selected_seeds = sorted(seeds[map_id])
        if len(selected_seeds) != seed_pairs or seen_seeds.intersection(selected_seeds):
            raise ValueError("Each map needs its own exact independent seed blocks")
        seen_seeds.update(selected_seeds)
        values = np.empty((seed_pairs, len(names), 2), np.float64)
        for opponent_index, name in enumerate(names):
            cell_rows: list[Mapping[str, Any]] = []
            for seed_index, seed_id in enumerate(selected_seeds):
                for end in (0, 1):
                    try:
                        row = indexed[map_id, seed_id, name, end]
                    except KeyError as error:
                        raise ValueError(
                            "A dependent validation block is incomplete"
                        ) from error
                    values[seed_index, opponent_index, end] = _score(
                        row["system_game_score"]
                    )
                    cell_rows.append(row)
            scores = values[:, opponent_index].reshape(-1)
            cell: Record = {
                "map_id": map_id,
                "opponent": name,
                "games": len(scores),
                "seed_pairs": seed_pairs,
                "score": float(scores.mean()),
                "wins": int(np.count_nonzero(scores == 1)),
                "draws": int(np.count_nonzero(scores == 0.5)),
                "losses": int(np.count_nonzero(scores == 0)),
            }
            for column in (
                "episode_length",
                *_SCORE_COLUMNS,
                *(_KILL_COLUMNS if actual_kills else ()),
            ):
                present = [row.get(column) for row in cell_rows]
                cell[f"mean_{column}"] = (
                    float(np.mean(np.asarray(present, np.float64)))
                    if all(x is not None for x in present)
                    else None
                )
            cells.append(cell)
        blocks.append(values.mean(axis=(1, 2)))
    low, high = _bootstrap(blocks, draws=bootstrap_draws, seed=bootstrap_seed)
    summary = {
        "complete": True,
        "score": float(np.mean([values.mean() for values in blocks])),
        "ci_low": low,
        "ci_high": high,
        "games": expected_count,
        "independent_blocks": len(map_ids) * seed_pairs,
        "cells": cells,
        "uncertainty": (
            "Conditional game-sampling interval; shared opponents and spawn ends "
            "stay together"
        ),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
    }
    return _validation_sampling_interval(summary, sampling_evidence, seed_pairs)


def _validation_sampling_interval(
    summary: Record, evidence: Mapping[str, Any] | None, seed_pairs: int
) -> Record:
    """Label fixed-System intervals using the shared sampling-evidence contract.

    None preserves historical numerical fields and meaning. Supplied evidence is
    checked against the completed games. independent_blocks and declared_blocks
    retain the historical declared-group count. The separate supported count is
    zero for deterministic conditions and None when unknown. Repeated calls
    preserve the original declared count.
    Unsupported sampling or fewer than two seed pairs per estimator stratum
    suppresses public ci_low/high. The former calculation remains only as
    explicitly conditional bounds, with its assumption stated. This does not
    establish that a provider obeys the declared mechanism.
    """
    if evidence is None:
        return summary
    from marl_battlegrounds.evaluation.sampling_evidence import (
        validate_sampling_evidence,
    )

    checked = validate_sampling_evidence(evidence, completed_games=summary["games"])
    status = checked["interval_status"]
    if status == "Available" and seed_pairs < 2:
        status = "Unavailable: Each validation stratum needs at least two seed pairs"
    result = {
        **summary,
        "declared_blocks": summary.get(
            "declared_blocks", summary["independent_blocks"]
        ),
        "independent_blocks": summary.get(
            "declared_blocks", summary["independent_blocks"]
        ),
        "supported_independent_sampling_units": checked[
            "supported_independent_sampling_units"
        ],
        "sampling_evidence": checked,
        "interval_status": status,
        "confidence": 0.95,
        "sampling_unit": "Paired game-seed blocks within declared validation strata",
        "interval_method": "Stratified percentile bootstrap for fixed Systems",
    }
    if status != "Available":
        result.update(
            conditional_ci_low=result.get("conditional_ci_low", result["ci_low"]),
            conditional_ci_high=result.get("conditional_ci_high", result["ci_high"]),
            conditional_interval_assumption=(
                "Assumes independent random game-seed blocks within every "
                "validation stratum; that assumption is unsupported here"
            ),
            ci_low=None,
            ci_high=None,
            uncertainty=f"Fixed-System game interval {status.lower()}",
        )
    return result


def _independent_validation_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    maps: Sequence[int],
    opponents: Sequence[str],
    seed_pairs: int,
    bootstrap_draws: int,
    bootstrap_seed: int,
    actual_kills: bool = False,
) -> Record:
    """Keep spawn pairs together while resampling independent opponents separately.

    mean_kill_difference is Team A kills minus Team B kills per game. With
    actual_kills=True the kills are the recorded team_a_kills/team_b_kills
    columns; with False they are the native Team A and Team B scores, which
    equal kills only under one point per death (results saved before the Red
    Zone rule). Every game must supply finite values. Maps and opponents have
    equal weights. The interval concerns these fixed actors only.
    """
    if not opponents or len(set(opponents)) != len(opponents):
        raise ValueError("Validation needs distinct nonempty opponents")
    if any(row.get("opponent") not in opponents for row in rows):
        raise ValueError("Validation contains an undeclared opponent")
    team_a, team_b = _KILL_COLUMNS if actual_kills else _SCORE_COLUMNS
    cells: list[Record] = []
    opponent_summaries: list[Record] = []
    blocks: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for opponent in opponents:
        selected = [row for row in rows if row.get("opponent") == opponent]
        checked = summarize_validation(
            selected,
            maps=maps,
            opponents=(opponent,),
            seed_pairs=seed_pairs,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed,
            actual_kills=actual_kills,
        )
        for cell in checked["cells"]:
            map_rows = [row for row in selected if row["map_id"] == cell["map_id"]]
            differences: list[float] = []
            pairs: dict[int, list[float]] = {}
            for row in map_rows:
                values = [row.get(team_a), row.get(team_b)]
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in values
                ):
                    raise ValueError(
                        "New validation needs finite recorded Team A and Team B kills"
                        if actual_kills
                        else "New validation needs finite native Team A and Team B "
                        "scores"
                    )
                differences.append(
                    float(cast(float, values[0])) - float(cast(float, values[1]))
                )
                pairs.setdefault(row["seed_id"], []).append(
                    _score(row["system_game_score"])
                )
            cell["mean_kill_difference"] = float(np.mean(differences))
            cells.append(cell)
            blocks.append(
                np.asarray([np.mean(pairs[seed]) for seed in sorted(pairs)], np.float64)
            )
        opponent_cells = [cell for cell in cells if cell["opponent"] == opponent]
        opponent_summaries.append(
            {
                "opponent": opponent,
                "score": checked["score"],
                "mean_kill_difference": float(
                    np.mean([cell["mean_kill_difference"] for cell in opponent_cells])
                ),
                "games": checked["games"],
            }
        )
    low, high = _bootstrap(blocks, draws=bootstrap_draws, seed=bootstrap_seed)
    return {
        "complete": True,
        "score": sum(_score(row["system_game_score"]) for row in rows) / len(rows),
        "mean_kill_difference": float(
            np.mean([cell["mean_kill_difference"] for cell in cells])
        ),
        "ci_low": low,
        "ci_high": high,
        "games": len(rows),
        "independent_blocks": len(maps) * len(opponents) * seed_pairs,
        "cells": cells,
        "opponents": opponent_summaries,
        "uncertainty": (
            "Conditional game-sampling interval; spawn ends stay paired "
            "and opponents are sampled independently"
        ),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
    }


def _mean_point_margin(row: Mapping[str, Any]) -> float:
    """Average Team A minus Team B points over complete map/opponent cells.

    Each cell must contain finite mean_team_a_score and mean_team_b_score.
    Validation gives every cell equal weight. Missing points are an error;
    wins and kills cannot stand in for them. Historical summaries stay unchanged.
    """
    cells = row.get("cells")
    if not isinstance(cells, (list, tuple)) or not cells:
        raise ValueError("Point-margin selection needs saved points in every cell")
    cells = cast(Sequence[object], cells)
    margins: list[float] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("Point-margin selection needs saved points in every cell")
        cell = cast(Mapping[str, Any], cell)
        points = [cell.get(f"mean_team_{team}_score") for team in ("a", "b")]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in points
        ):
            raise ValueError(
                "Point-margin selection needs finite saved points in every cell"
            )
        points = cast(list[float], points)
        margin = float(points[0]) - float(points[1])
        if not math.isfinite(margin):
            raise ValueError("Point-margin selection needs a finite point difference")
        margins.append(margin / len(cells))
    return math.fsum(margins)


def _selection_key(
    row: Mapping[str, Any], *, rule: str = "saved"
) -> tuple[float, float, int, str]:
    """Order by point margin, or reproduce the saved historical score rule.

    point_margin uses only mean points, then earlier steps and checkpoint ID.
    saved keeps schema 1 score ties and schema 2 score/kill ties unchanged.
    """
    if rule not in ("point_margin", "saved"):
        raise ValueError("Selection rule must be point_margin or saved")
    difference = 0.0
    if rule == "point_margin":
        primary = _mean_point_margin(row)
    else:
        primary = float(row["score"])
        if row.get("selection_schema_version", 1) == 2:
            value = row.get("mean_kill_difference")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    "Saved checkpoint selection needs a finite mean kill difference"
                )
            difference = float(value)
    return (
        -primary,
        -difference,
        int(row["env_steps"]),
        str(row["checkpoint_id"]),
    )


def _candidates(
    results: Sequence[Mapping[str, Any]], *, rule: str = "saved"
) -> dict[str, Record]:
    """Validate complete unique noninitial checkpoint summaries for selection.

    A row with no experience is dropped. A row marked ``method == "qmix"``
    or ``"pqn_vdn"`` must carry a nonnegative integer optimizer_steps
    (missing, negative or Boolean values raise ValueError) and is dropped at
    zero, so QMIX warmup and PQN-VDN initial-collection actors that never
    learned are never selected. Compatible PPO, QMIX and PQN-VDN rows (same
    frozen panel, maps, ordered rosters and scoring protocol) may be selected
    together. Missing roster fields retain the historical canonical 5v5 meaning.
    Rows must also share one scoring rule: the same recorded red_zone_depth,
    or none (results saved before the Red Zone rule). Mixing them raises
    ValueError because their points and kill differences are not comparable.
    rule chooses the ranking fields: point_margin for new choices, saved for
    historical rules.
    """
    selected: dict[str, Record] = {}
    for result in results:
        identifier = result.get("checkpoint_id")
        if not isinstance(identifier, str) or not identifier or identifier in selected:
            raise ValueError("Selection needs distinct nonempty checkpoint identities")
        if result.get("complete") is not True:
            raise ValueError("Incomplete validation cannot select a checkpoint")
        steps = _integer(result.get("env_steps"), "env_steps")
        value = result.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Selection score is missing")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Selection score must be finite and between zero and one")
        trained = bool(steps)
        if result.get("method") in ("qmix", "pqn_vdn"):
            trained &= bool(_integer(result.get("optimizer_steps"), "optimizer_steps"))
        if trained:
            selected[identifier] = dict(result)
    if not selected:
        raise ValueError("No eligible trained checkpoint is available")
    from marl_battlegrounds.training.validation import validation_conditions

    conditions = [validation_conditions(row) for row in selected.values()]
    if any(row != conditions[0] for row in conditions[1:]):
        raise ValueError(
            "Selection cannot mix panels, maps, team rosters, "
            "deployed partners or scoring conditions"
        )
    for row in selected.values():
        _selection_key(row, rule=rule)
    return selected


def confirmation_candidates(
    routine_results: Sequence[Mapping[str, Any]],
    *,
    final_checkpoint_id: str,
    rule: str = "point_margin",
) -> tuple[str, ...]:
    """Return the best two routine checkpoint IDs, then final when distinct.

    routine_results must contain complete, unique fixed-panel summaries with
    checkpoint_id, env_steps and score. The default point_margin rule ranks only
    mean Team A minus Team B points from the saved cells. Every cell needs finite
    mean_team_a_score and mean_team_b_score. Equal margins use earlier steps,
    then checkpoint ID; wins and kills do not break ties. rule='saved' reproduces
    a historical panel's original score and tie rules.

    Initialization is ignored. QMIX and PQN-VDN rows need a nonnegative integer
    optimizer_steps and are skipped at zero. The final checkpoint must be
    present. Incomplete, mismatched or missing evidence raises ValueError.
    No files change.
    """
    candidates = _candidates(routine_results, rule=rule)
    if final_checkpoint_id not in candidates:
        raise ValueError("The final checkpoint lacks complete routine validation")
    ordered = sorted(
        candidates,
        key=lambda key: _selection_key(candidates[key], rule=rule),
    )[:2]
    if final_checkpoint_id not in ordered:
        ordered.append(final_checkpoint_id)
    return tuple(ordered)


def select_checkpoint(
    confirmation_results: Sequence[Mapping[str, Any]], *, rule: str = "point_margin"
) -> Record:
    """Choose a fresh-confirmation winner by mean point margin.

    Each input uses the same summary contract as confirmation_candidates and must
    have purpose='confirmation'. The default point_margin rule uses only mean
    Team A minus Team B points; ties use earlier steps, then checkpoint ID.
    rule='saved' reproduces a historical panel's original score and tie rules.
    Return a new copy of the winning summary. The caller checks that every
    scheduled candidate was confirmed. Empty, incomplete, mixed-panel or
    wrong-purpose records raise ValueError. No I/O occurs.
    """
    if any(row.get("purpose") != "confirmation" for row in confirmation_results):
        raise ValueError("Selection requires fresh confirmation results")
    candidates = _candidates(confirmation_results, rule=rule)
    return dict(
        min(
            candidates.values(),
            key=lambda row: _selection_key(row, rule=rule),
        )
    )


def slot_planning_blocks(
    *, effect: float = 0.1, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Return the normal-approximation block count using the variance bound one.

    effect is a positive expected-score difference; power and alpha lie in (0,1).
    The default gives 785 blocks before balancing across maps. This is a planning
    approximation, not a distribution-free power guarantee. Invalid settings raise
    ValueError. The host calculation has no randomness or file effects.
    """
    if not 0 < effect <= 1 or not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid slot diagnostic planning settings")
    normal = NormalDist()
    return math.ceil(
        ((normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)) / effect) ** 2
    )


def _effect_summary(
    blocks: Sequence[np.ndarray[Any, np.dtype[np.float64]]], *, seed: int, draws: int
) -> Record:
    """Compute an equal-map mean effect and its stratified independent-block error.

    blocks contains one float64 vector per map, with at least two independent
    blocks per vector. A zero estimated variance leaves inference unavailable.
    The normal p-value is approximate and the bootstrap interval is descriptive.
    """
    effect = float(np.mean([value.mean() for value in blocks]))
    variance = (
        sum(float(value.var(ddof=1)) / len(value) for value in blocks)
        / len(blocks) ** 2
    )
    error = math.sqrt(variance)
    if error == 0:
        return {
            "effect": effect,
            "standard_error": 0.0,
            "p_value": None,
            "ci_low": None,
            "ci_high": None,
            "inference": "Unavailable: zero estimated variance",
        }
    low, high = _bootstrap(blocks, draws=draws, seed=seed)
    return {
        "effect": effect,
        "standard_error": error,
        "p_value": math.erfc(abs(effect / error) / math.sqrt(2)),
        "ci_low": low,
        "ci_high": high,
        "inference": (
            "Approximate normal mean test; independent blocks; "
            "descriptive bootstrap interval"
        ),
    }


def summarize_slot_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    *,
    maps: Sequence[int],
    seed_blocks: int = 160,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 19_044_002,
) -> Record:
    """Compare a fixed focal model across slot blocks while keeping its side fixed.

    Rows require map_id, seed_id, focal_team (0/1), spawn_locations (0/1) and
    evaluation's system_game_score. Four games form each independent map/seed block.
    Team B's focal score is one minus evaluation's Team A score. maps are distinct and
    equally weighted; seed_blocks is at least two. Return primary side-balanced and both
    physical-side effects, normal p-values, Holm-adjusted side p-values and counts. The
    independent seeds, unchanged physical conditions and numerical equivalence of
    checkpoints are caller evidence obligations. Missing, duplicate or malformed
    conditions raise ValueError. This host-only helper changes no files or inputs.
    """
    _integer(seed_blocks, "seed_blocks", minimum=2)
    map_ids = tuple(maps)
    if not map_ids or len(set(map_ids)) != len(map_ids):
        raise ValueError("Slot diagnostic maps must be nonempty and distinct")
    if len(rows) != len(map_ids) * seed_blocks * 4:
        raise ValueError("Slot diagnostic has missing or extra games")
    indexed: dict[tuple[int, int, int, int], float] = {}
    seeds: dict[int, set[int]] = {map_id: set() for map_id in map_ids}
    for row in rows:
        map_id = _integer(row.get("map_id"), "map_id")
        seed = _integer(row.get("seed_id"), "seed_id")
        team = _integer(row.get("focal_team"), "focal_team")
        end = _integer(row.get("spawn_locations"), "spawn_locations")
        if map_id not in seeds or team not in (0, 1) or end not in (0, 1):
            raise ValueError("Invalid slot diagnostic condition")
        key = (map_id, seed, team, end)
        if key in indexed:
            raise ValueError("Slot diagnostic repeats a condition")
        score = _score(row.get("system_game_score"))
        indexed[key] = score if team == 0 else 1 - score
        seeds[map_id].add(seed)
    seen: set[int] = set()
    side_blocks: tuple[list[np.ndarray[Any, np.dtype[np.float64]]], ...] = ([], [])
    balanced: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for map_id in map_ids:
        values = np.empty((seed_blocks, 2), np.float64)
        selected = sorted(seeds[map_id])
        if len(selected) != seed_blocks or seen.intersection(selected):
            raise ValueError("Slot seed blocks must be complete and map-distinct")
        seen.update(selected)
        for i, seed in enumerate(selected):
            for side in (0, 1):
                try:
                    values[i, side] = (
                        indexed[map_id, seed, 0, side]
                        - indexed[map_id, seed, 1, 1 - side]
                    )
                except KeyError as error:
                    raise ValueError("Slot diagnostic block is incomplete") from error
        for side in (0, 1):
            side_blocks[side].append(values[:, side])
        balanced.append(values.mean(axis=1))
    primary = _effect_summary(balanced, seed=bootstrap_seed, draws=bootstrap_draws)
    sides = [
        _effect_summary(blocks, seed=bootstrap_seed + side + 1, draws=bootstrap_draws)
        for side, blocks in enumerate(side_blocks)
    ]
    available = sorted(
        (float(row["p_value"]), i)
        for i, row in enumerate(sides)
        if row["p_value"] is not None
    )
    previous = 0.0
    for rank, (value, index) in enumerate(available):
        previous = max(previous, min(1.0, (2 - rank) * value))
        sides[index]["holm_p_value"] = previous
    for side, row in enumerate(sides):
        row["physical_side"] = side
        row.setdefault("holm_p_value", None)
    return {
        "complete": True,
        "games": len(rows),
        "independent_blocks": len(map_ids) * seed_blocks,
        "primary": primary,
        "sides": sides,
        "alpha": 0.05,
        "systematic_effect_detected": bool(
            (primary["p_value"] is not None and primary["p_value"] < 0.05)
            or any(
                row["holm_p_value"] is not None and row["holm_p_value"] < 0.05
                for row in sides
            )
        ),
        "scope": (
            "These frozen models and validation maps only; no universal fairness claim"
        ),
    }


def _read_json(path: Path, default: object) -> Any:  # noqa: ANN401
    """Read one JSON file or return default when absent; malformed JSON raises."""
    return json.loads(path.read_text()) if path.exists() else default


def _read_rows(path: Path) -> list[Record]:
    """Read committed newline-delimited JSON objects; reject a partial final row."""
    if not path.exists():
        return []
    if (
        path.name == "training_updates.jsonl"
        and (path.parent / "latest_checkpoint.json").exists()
    ):
        return [
            {key: value for key, value in row.items() if key != "acceptance"}
            for row in _export_update_rows(path.parent, include_archives=False)
            if row["acceptance"] == "Accepted"
        ]
    text = path.read_text()
    if text and not text.endswith("\n"):
        raise ValueError(f"Incomplete committed log row: {path.name}")
    rows = [json.loads(line) for line in text.splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected object rows in {path.name}")
    return cast(list[Record], rows)


def _atomic_text(path: Path, content: str) -> None:
    """Replace one report atomically on its filesystem; fsync before publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_events(path: Path) -> list[Record]:
    """Use the shared append-only reader, keeping malformed attempt evidence."""
    from marl_battlegrounds.training._run_io import read_run_events

    return read_run_events(path)


def _exposure_history(directory: Path) -> list[Record]:
    """Read accepted and superseded marginal rows using the status checkpoint.

    No event step or newest-row heuristic grants acceptance. Missing status,
    invalid checkpoint evidence or an active recovery marker withholds every
    active row. Raw complete events remain labelled uncommitted/superseded with
    their own attempts and costs. Historical runs without events return [].
    Relocated runs may resolve the same checkpoint basename under their own
    checkpoints folder; its checked description must retain the saved run ID.
    This reads descriptions and event JSON only, never model arrays.
    """
    from marl_battlegrounds.training._checkpoint_metadata import (
        read_checkpoint_description,
    )
    from marl_battlegrounds.training._run_io import checked_exposure_snapshots

    events = _read_events(directory / "run_events.jsonl")
    snapshots = [event for event in events if event.get("event") == "exposure_snapshot"]
    if not snapshots:
        return []
    accepted: dict[str, Record] = {}
    reason: str | None = None
    try:
        if (directory / "checkpoint_recovery.json").exists():
            raise ValueError("Checkpoint recovery is in progress")
        status = _read_json(directory / "status.json", None)
        details = _read_json(directory / "run_details.json", None)
        if not isinstance(status, dict) or not isinstance(details, dict):
            raise ValueError("Run status or identity is missing")
        status, details = cast(Record, status), cast(Record, details)
        if status.get("run_id") != details.get("run_id") or not isinstance(
            status.get("latest_checkpoint"), str
        ):
            raise ValueError("Run status has no matching saved checkpoint")
        path = Path(status["latest_checkpoint"])
        if not path.is_absolute():
            path = directory / path
        if not path.resolve().is_relative_to(directory.resolve()):
            path = directory / "checkpoints" / path.name
        checkpoint = read_checkpoint_description(path)
        metadata = checkpoint["metadata"]
        if checkpoint["kind"] != "learner" or metadata["run_id"] != details["run_id"]:
            raise ValueError("Exposure checkpoint belongs to a different run")
        accepted = checked_exposure_snapshots(
            metadata["host_state"].get("exposure_snapshots", {}),
            events,
            run_id=details["run_id"],
            env_steps=checkpoint["counters"]["env_steps"],
        )
    except (ValueError, OSError, TypeError, KeyError) as error:
        reason = str(error)
    accepted_ids = {event["event_id"] for event in accepted.values()}
    rows: list[Record] = []
    for event in snapshots:
        active = event.get("event_id") in accepted_ids
        event_rows = event.get("rows")
        if not isinstance(event_rows, list):
            continue
        for row in cast(list[Any], event_rows):
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    **{
                        key: value
                        for key, value in event.items()
                        if key not in ("rows", "event")
                    },
                    **cast(Record, row),
                    "accepted": active,
                    "evidence_status": "accepted"
                    if active
                    else "uncommitted_or_superseded",
                    "unavailable_reason": None
                    if active
                    else reason or "Event is not referenced by the active checkpoint",
                }
            )
    return rows


def _csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize scalar report rows with a stable first-seen union of columns."""
    columns = tuple(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _treatment(config: Mapping[str, Any]) -> str:
    """Name the enabled curriculum/shaping options; absent options mean Plain."""
    shaping = (
        "Score Delta" if config.get("shaping_mode") == "score_delta" else "Shaping"
    )
    return (
        "+".join(
            name
            for key, name in (("curriculum", "Curriculum"), ("shaping", shaping))
            if config.get(key)
        )
        or "Plain"
    )


def _phase_costs(events: Sequence[Record], details: Record) -> Record:
    """Sum saved finite phase seconds across all attempts without new timing.

    Count only the phase owner's completed event, not nested evaluation segments.
    Missing or invalid times remain unavailable. Initial setup in run_details is
    a fallback only when no attempt setup event has a usable time; it is never
    added to that same event. Interrupted work may have no completed measurement.
    """
    costs: Record = {}
    for phase, event, field in (
        ("Setup", "attempt_started", "setup_seconds"),
        ("Validation", "validation_complete", "seconds"),
        ("Random Validation", "random_validation_complete", "seconds"),
        ("Checkpoint", "checkpoint_saved", "seconds"),
        ("Report", "reports_written", "seconds"),
        ("Slot Diagnostic", "slot_diagnostic_finished", "seconds"),
    ):
        values = [
            float(row[field])
            for row in events
            if row.get("event") == event
            and type(row.get(field)) in (float, int)
            and math.isfinite(row[field])
            and row[field] >= 0
        ]
        source = "run_events.jsonl"
        initial = details.get("initial_setup_seconds")
        if (
            phase == "Setup"
            and not values
            and isinstance(initial, (float, int))
            and not isinstance(initial, bool)
            and math.isfinite(initial)
            and initial >= 0
        ):
            values = [float(initial)]
            source = "run_details.json (initial setup only)"
        costs[phase] = {
            "seconds": math.fsum(values) if values else None,
            "records": len(values),
            "source": source,
        }
    return costs


def _settings(config: Mapping[str, Any]) -> Record:
    """Return the settings block of a saved run config for its method.

    config is the parsed config.json. Returns the block that
    ``methods.method_settings_field`` names: config["qmix"] for "qmix",
    config["pqn"] for "pqn_vdn" and config["ppo"] otherwise (a missing
    method means MAPPO); a missing or non-object block gives an empty dict,
    so callers apply their own defaults. An unknown method raises ValueError.
    """
    from marl_battlegrounds.baselines.methods import method_settings_field

    block = config.get(method_settings_field(config.get("method", "mappo")), {})
    return cast(Record, block) if isinstance(block, dict) else {}


def _red_zone_label(depth: object) -> str:
    """Format a saved depth in map units; absent values stay explicitly unknown.

    This host report helper reads no defaults or files. None means no value was
    recorded; it is distinct from an explicit zero. Invalid saved depths fail.
    """
    if depth is None:
        return "Unknown (not recorded)"
    if (
        isinstance(depth, bool)
        or not isinstance(depth, (int, float))
        or not math.isfinite(depth)
        or depth < 0
    ):
        raise ValueError("Saved Red Zone depth must be a finite nonnegative number")
    return f"{float(depth)} map units"


def _summary(directory: Path) -> Record:
    """Read run facts, withholding active results while checkpoint recovery is open.

    An unfinished recovery can leave an older complete status and output files.
    Keep their paths as evidence, but do not label them as current results or load
    the possibly partly recovered update log. ``selection_unavailable_reason``
    copies the runner's saved reason for a run that ended without a selection.
    This reader changes no source file.
    """
    details = _read_json(directory / "run_details.json", None)
    if not isinstance(details, dict):
        raise ValueError(f"Missing run_details.json in {directory}")
    details = cast(Record, details)
    config = details.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("run_details.json config must contain an object")
    config = cast(Record, config)
    recovering = (directory / "checkpoint_recovery.json").exists()
    recovery = (
        _read_json(directory / "checkpoint_recovery.json", None) if recovering else None
    )
    updates = [] if recovering else _read_rows(directory / "training_updates.jsonl")
    events = _read_events(directory / "run_events.jsonl")
    results: object = (
        [] if recovering else _read_json(directory / "validation_results.json", [])
    )
    if isinstance(results, dict):
        results = cast(Record, results).get("results", [])
    if not isinstance(results, list):
        raise ValueError("validation_results.json must contain result objects")
    status = _read_json(directory / "status.json", {})
    if not isinstance(status, dict):
        raise ValueError("status.json must contain an object")
    status = cast(Record, status)
    return {
        "run_id": str(details.get("run_id", directory.name)),
        "run_dir": str(directory),
        "method": config.get("method", "Unavailable"),
        "treatment": _treatment(config),
        "shaping_mode": config.get("shaping_mode", "potential"),
        "input_scale": _settings(config).get("input_scale", 1.0),
        "spawn_frame": _settings(config).get("spawn_frame", "world"),
        "red_zone_depth": config.get("red_zone_depth"),
        "pinned_opponent": config.get("pinned_opponent"),
        "pinned_opponent_share": config.get("pinned_opponent_share", 0.0),
        "seed": config.get("seed"),
        "declared_env_steps": config.get("total_env_steps"),
        "details_path": str(directory / "run_details.json"),
        "schemas": details.get("schemas"),
        "initial_runtime": details.get("runtime"),
        "attempt_runtimes": [
            {
                "attempt_id": event.get("attempt_id"),
                "time_utc": event.get("time_utc"),
                "runtime": event.get("runtime"),
            }
            for event in events
            if event.get("event") == "attempt_started"
        ],
        "memory": None if recovering else status.get("memory"),
        "terminal_memory_history": [
            {
                "attempt_id": event.get("attempt_id"),
                "time_utc": event.get("time_utc"),
                "event": event["event"],
                "memory": event.get("memory"),
            }
            for event in events
            if event.get("event") in ("attempt_complete", "attempt_failed")
        ],
        "panel_digest": details.get("panel_digest"),
        "status": "recovering" if recovering else status.get("status", "incomplete"),
        "saved_status": status.get("status", "incomplete"),
        "recovery": recovery,
        "elapsed_seconds": None if recovering else status.get("elapsed_seconds"),
        "wall_seconds": None if recovering else status.get("wall_seconds"),
        "env_steps": None
        if recovering
        else updates[-1].get("env_steps", 0)
        if updates
        else 0,
        "validation_checkpoints": len(cast(list[object], results)),
        "validation_red_zone_depths": list(
            dict.fromkeys(
                cast(Record, row).get("red_zone_depth")
                for row in cast(list[object], results)
                if isinstance(row, dict)
            )
        ),
        "recorded_failures": sum(
            row.get("event") in ("failure", "attempt_failed") for row in events
        ),
        "incomplete_events": sum(
            row.get("event") == "incomplete_event" for row in events
        ),
        "phase_costs": _phase_costs(events, details),
        "selection": None
        if recovering
        else _read_json(directory / "selection.json", None),
        "final_actor": None if recovering else status.get("final_actor"),
        "selected_actor": None if recovering else status.get("selected_actor"),
        "selection_unavailable_reason": None
        if recovering
        else status.get("selection_unavailable_reason"),
        "latest_update": updates[-1] if updates else None,
        "exposure": None
        if recovering
        else _read_json(directory / "exposure.json", None),
        "slot_diagnostic": None
        if recovering
        else _read_json(directory / "slot_diagnostic.json", None),
        **(
            {
                # The first T rounds (T * B transitions) are never learned;
                # a run saved before round T has generated fewer.
                "initial_transitions_not_learned": min(
                    _settings(config).get("rollout_length", 128)
                    * config.get("num_envs", 0),
                    updates[-1].get("env_steps", 0) if updates else 0,
                )
            }
            if config.get("method") == "pqn_vdn"
            else {}
        ),
    }


def _runtime_text(runtime: object) -> str:
    """Describe saved evaluation runtime facts; missing old fields stay unknown."""
    if not isinstance(runtime, dict):
        return "Unknown; no runtime record was saved."
    record = cast(Record, runtime)
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        return "Unknown; no M8 runtime provenance was saved."
    facts = cast(Record, provenance)
    return (
        f"Backend: {facts.get('backend', 'Unknown')}; "
        f"Device: {facts.get('device', 'Unknown')}; "
        f"Precision: {facts.get('precision', 'Unknown')}; "
        f"Batch Shape: {facts.get('batch_shape', 'Unknown')}."
    )


def _memory_lines(memory: object, *, label: str) -> list[str]:
    """Format one saved terminal memory record with its original limits and errors.

    label says which recorded status or attempt owns this snapshot. RAM uses bytes;
    allocator counters keep their saved names and values. Missing records/counters
    remain unknown. This formats existing facts and queries no process or device.
    """
    if not isinstance(memory, dict):
        return [f"{label}: Unknown; no memory snapshot was saved.", ""]
    record = cast(Record, memory)
    ram = record.get("process_peak_ram_bytes")
    allocator = record.get("jax_allocator")
    return [
        f"{label}:",
        f"Process Lifetime Peak RAM: {ram} bytes."
        if ram is not None
        else "Process Lifetime Peak RAM: Unknown.",
        f"JAX Allocator Counters: `{json.dumps(allocator, sort_keys=True)}`."
        if allocator is not None
        else "JAX Allocator Counters: Unknown; no counters were available.",
        f"Measurement Scope: {record.get('scope', 'Unknown')}",
        f"Measurement Read Errors: `{json.dumps(record.get('errors'))}`.",
        "",
    ]


def _no_selection_reason(summary: Record) -> str:
    """Say why a finished run has no selected actor and what to load instead.

    Parameters
    ----------
    summary : dict
        One record from ``_summary`` for a run that saved a final actor but no
        selected actor. ``selection_unavailable_reason`` is the runner's saved
        reason or absent; ``validation_checkpoints`` counts saved validation
        results.

    Returns
    -------
    str
        One or two plain sentences. A saved reason wins. A run with no
        validation results says that validation was off and names the two
        settings that turn it on. Otherwise it says that no selection was saved.
        Every case points to the final actor, which the summary shows next.
    """
    reason = summary.get("selection_unavailable_reason")
    if isinstance(reason, str) and reason:
        return f"{reason}, so no best checkpoint was chosen. Load the final actor."
    if not summary.get("validation_checkpoints"):
        return (
            "This run had no validation, so no best checkpoint was chosen. "
            "Load the final actor. To choose a best checkpoint, set "
            "validation_opponents or validation_panel."
        )
    return "No selection was saved. Load the final actor."


def _write_summary(summaries: Sequence[Record], destination: Path) -> Path:
    """Atomically write factual Markdown from already-read per-run summaries.

    A finished run without a selected actor says why (no validation, or the
    runner's saved reason) and points to its final actor; only an unfinished or
    recovering run keeps "Unavailable in the saved active status".
    A QMIX run reports optimizer steps, used TD pairs and replay rows per game.
    A PQN-VDN run reports optimizer steps, learning blocks, used TD pairs with
    the pairs starting on kept rows, and the initial random transitions that
    were generated but never learned (the first rollout_length rounds, or
    fewer when the run has not reached them). A PPO run reports used policy and value
    samples.
    """
    lines = [
        "# Training Run Summary",
        "",
        "This report contains measured records. Useful learning and broader "
        "competence still need an evidence review.",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['run_id']}",
                "",
                f"Method: {summary['method']}. Treatment: {summary['treatment']}. "
                f"Training seed: {summary['seed']}.",
                f"Shaping Mode: {summary['shaping_mode']}. "
                f"Input Scale: {summary['input_scale']}. "
                f"Spawn Frame: {summary['spawn_frame']}.",
                "Training Red Zone Depth: "
                f"{_red_zone_label(summary['red_zone_depth'])}.",
                "Recorded Validation Red Zone Depths: "
                + (
                    "; ".join(
                        _red_zone_label(depth)
                        for depth in summary["validation_red_zone_depths"]
                    )
                    or "No recorded validation"
                )
                + ".",
                *(
                    [
                        f"Pinned Opponent: {summary['pinned_opponent']} in "
                        f"{summary['pinned_opponent_share']} of new games, a "
                        "training opponent; its evidence is in exposure.json."
                    ]
                    if summary["pinned_opponent"] is not None
                    else []
                ),
                "Declared budget: "
                f"{summary['declared_env_steps']} environment transitions.",
                "Exact settings, source identity (`source`), content identity "
                "(`content_binding`), schema versions (`schemas`), initial runtime "
                "(`runtime`), dependency versions and frozen panel identity "
                f"(`panel_digest`): [run_details.json](<{summary['details_path']}>).",
                f"Frozen panel identity: `{summary['panel_digest']}`. "
                "None means no panel was declared.",
                "",
                f"Status: {summary['status']}. Recorded environment transitions: "
                f"{summary['env_steps']}.",
                f"Validation records: {summary['validation_checkpoints']}. "
                f"Recorded failures: {summary['recorded_failures']}.",
                f"Incomplete event records: {summary['incomplete_events']}. "
                "Their work is unknown.",
                f"Original evidence: `{summary['run_dir']}`.",
                f"Recorded Active Time: {summary['elapsed_seconds']} seconds. "
                f"Wall duration including pauses: {summary['wall_seconds']} seconds. "
                "None means the measurement is unavailable.",
                "Recorded active time carries forward the larger saved checkpoint "
                "or same-run status value, then adds this attempt. Interrupted "
                "work without a durable timing record may be missing. Wall duration "
                "includes the gaps between attempts.",
                "",
            ]
        )
        lines.extend(
            [
                "Schema Versions: "
                + (
                    f"`{json.dumps(summary['schemas'], sort_keys=True)}`."
                    if summary["schemas"] is not None
                    else "Unknown; no schema record was saved."
                ),
                "Initial Runtime: " + _runtime_text(summary["initial_runtime"]),
                "Declared device environment values remain in `runtime` in "
                "run_details.json. These selectors do not verify a physical GPU's "
                "UUID or PCI bus; the frozen launch package owns that check.",
                "",
            ]
        )
        if summary["attempt_runtimes"]:
            lines.append("Recorded Runtime By Attempt:")
            lines.append("")
            for attempt in summary["attempt_runtimes"]:
                lines.append(
                    f"- `{attempt['attempt_id']}`: " + _runtime_text(attempt["runtime"])
                )
            lines.append("")
        lines.extend(
            [
                "Latest Status Memory: Withheld while checkpoint recovery "
                "is unfinished.",
                "",
            ]
            if summary["status"] == "recovering"
            else _memory_lines(summary["memory"], label="Latest Status Memory")
        )
        prior_memory = [
            record
            for record in summary["terminal_memory_history"]
            if record["memory"] != summary["memory"]
        ]
        if prior_memory:
            lines.extend(
                [
                    "Earlier terminal snapshots are historical measurements, "
                    "including failed or abandoned attempts. They are not current "
                    "memory use and their peaks must not be added together.",
                    "",
                ]
            )
            for record in prior_memory:
                lines.extend(
                    _memory_lines(
                        record["memory"],
                        label=f"Recorded {record['event']} / {record['attempt_id']}",
                    )
                )
        if summary["status"] == "recovering":
            lines.extend(
                [
                    "Checkpoint recovery is unfinished. The saved status before "
                    f"recovery was {summary['saved_status']}; it does not describe "
                    "the active attempt. Prior update, validation, actor, selection "
                    "and exposure files are preserved but withheld from this report.",
                    "Resume the checkpoint named in "
                    f"[checkpoint_recovery.json](<{summary['run_dir']}"
                    "/checkpoint_recovery.json>).",
                    "",
                ]
            )
        decision = summary.get("selection_decision")
        if decision is not None:
            original = summary.get("original_selection")
            original_id = (
                cast(Record, original).get("checkpoint_id")
                if isinstance(original, dict)
                else None
            )
            lines.extend(
                [
                    "Selection Source: Separate declared decision "
                    f"[{decision['name']}](<{decision['path']}>) "
                    f"(`{decision['decision_id']}`).",
                    "Original Saved Selection: "
                    f"[selection.json](<{summary['original_selection_path']}>); "
                    f"checkpoint `{original_id or 'Unavailable'}`. "
                    "The original record remains unchanged.",
                    "",
                ]
            )
        selected = summary["selection"]
        if isinstance(selected, dict):
            selection = cast(Record, selected)
            lines.extend(
                [
                    "Selected checkpoint: "
                    f"`{selection.get('checkpoint_id', 'Unavailable')}`. "
                    f"Confirmation score: {selection.get('score', 'Unavailable')}. "
                    "Confirmation Red Zone Depth: "
                    f"{_red_zone_label(selection.get('red_zone_depth'))}.",
                    "",
                ]
            )
        for key, label in (
            (
                "final_actor",
                "Final Actor (Extra Result)" if decision else "Final Actor",
            ),
            ("selected_actor", "Selected Actor"),
        ):
            actor = summary[key]
            lines.extend(
                [
                    f"{label}: [{actor}](<{actor}>)."
                    if isinstance(actor, str)
                    else f"{label}: None. {_no_selection_reason(summary)}"
                    if key == "selected_actor"
                    and isinstance(summary["final_actor"], str)
                    else f"{label}: Unavailable in the saved active status.",
                    "",
                ]
            )
        load_path = summary["selected_actor"] or summary["final_actor"]
        if isinstance(load_path, str):
            lines.extend(
                [
                    "Load this saved actor for evaluation:",
                    "",
                    "```python",
                    "from marl_battlegrounds import training",
                    f"system = training.load_system({load_path!r})",
                    "```",
                    "",
                ]
            )
        update = summary["latest_update"]
        if isinstance(update, dict):
            latest = cast(Record, update)
            lines.extend(
                [
                    "Recorded Active Time At The Last Update: "
                    f"{latest.get('elapsed_seconds', 'Unavailable')} seconds. "
                    "Collection/update time: "
                    f"{latest.get('training_seconds', 'Unavailable')} seconds.",
                    *(
                        (
                            "Optimizer steps: "
                            f"{latest.get('completed_updates', 'Unavailable')}. "
                            "Used TD pairs (repeats count again): "
                            f"{latest.get('used_td_pairs', 'Unavailable')}. "
                            "Replay rows per game: "
                            f"{latest.get('replay_rows_per_lane', 'Unavailable')}.",
                            "Losses, Q values, exploration, rewards and per-block "
                            "costs remain in learning_curve.csv.",
                        )
                        if summary["method"] == "qmix"
                        else (
                            "Optimizer steps: "
                            f"{latest.get('completed_updates', 'Unavailable')}. "
                            "Learning blocks: "
                            f"{latest.get('learning_blocks', 'Unavailable')}. "
                            "Used TD pairs (each epoch counts again): "
                            f"{latest.get('used_td_pairs', 'Unavailable')}, of "
                            "which pairs starting on kept rows: "
                            f"{latest.get('used_prefix_td_pairs', 'Unavailable')}. "
                            "Initial random transitions generated but never "
                            "learned: "
                            f"{summary.get('initial_transitions_not_learned')}.",
                            "Losses, Q values, exploration, learning rates, rewards "
                            "and per-block costs remain in learning_curve.csv.",
                        )
                        if summary["method"] == "pqn_vdn"
                        else (
                            "Used policy samples: "
                            f"{latest.get('used_policy_samples', 'Unavailable')}. "
                            "Used value samples: "
                            f"{latest.get('used_value_samples', 'Unavailable')}.",
                            "Losses, entropy, rewards and per-update costs remain in "
                            "learning_curve.csv.",
                        )
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                "Measured phase costs include completed records from all attempts, "
                "including retried work. Interrupted phases may have unmeasured work. "
                "These totals are not a breakdown of only the current branch.",
                "",
                "| Phase | Seconds | Recorded Measurements | Source |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for phase, cost in summary["phase_costs"].items():
            source_file = str(cost["source"]).split(" ", 1)[0]
            seconds = cost["seconds"] if cost["seconds"] is not None else "Unavailable"
            lines.append(
                f"| {phase} | {seconds} "
                f"| {cost['records']} | [{cost['source']}]"
                f"(<{summary['run_dir']}/{source_file}>) |"
            )
        lines.append("")
        if summary["exposure"] is not None:
            lines.extend(
                [
                    "Actual stage, map and opponent exposure, completed games and "
                    "unfinished games: "
                    f"[exposure.json]({summary['run_dir']}/exposure.json).",
                    "",
                ]
            )
        slot = summary["slot_diagnostic"]
        if isinstance(slot, dict):
            diagnostic = cast(Record, slot)
            detected = diagnostic.get("systematic_effect_detected")
            lines.extend(
                [
                    "Final slot diagnostic: "
                    + (
                        "A systematic effect was detected; a fairness claim is blocked."
                        if detected is True
                        else "No systematic effect was detected. "
                        "This is not proof of fairness."
                        if detected is False
                        else "Incomplete; no inference is available."
                    ),
                    "Exact scope and uncertainty: "
                    f"[slot_diagnostic.json]({summary['run_dir']}/slot_diagnostic.json).",
                    "",
                ]
            )
    lines.extend(
        [
            "Per-run validation intervals describe game sampling for fixed actors. "
            "They do not "
            "measure training-seed variation.",
            "Task reward and shaping remain separate. Confirmation scores are "
            "used for selection and are not held-out test results.",
            "Task reward is divided by active-agent samples; shaping is divided "
            "by real environment transitions.",
            "",
            "Exact settings, source/content identities and seeds remain in each "
            "run's run_details.json. Attempts and execution segments remain in "
            "run_events.jsonl.",
            "",
        ]
    )
    target = destination / "run_summary.md"
    _atomic_text(target, "\n".join(lines))
    return target


def refresh_summary(run_dirs: Sequence[str | Path], *, output_dir: str | Path) -> Path:
    """Refresh only factual Markdown after the runner publishes its final status.

    Parameters
    ----------
    run_dirs : sequence of paths
        Nonempty list of exact saved run directories, as accepted by analyze.
    output_dir : path
        Report directory. Only run_summary.md is atomically replaced. Original
        records and existing plots/CSVs are unchanged; no plotting library loads.

    Returns
    -------
    pathlib.Path
        Written run_summary.md. Completion uses saved status.json and requires
        no unfinished checkpoint_recovery.json marker.

    Raises
    ------
    ValueError, OSError
        Required records are missing, malformed or cannot be read or written.
    """
    if not run_dirs:
        raise ValueError("refresh_summary needs at least one exact run directory")
    return _write_summary(
        [_summary(Path(value).expanduser().resolve()) for value in run_dirs],
        Path(output_dir).expanduser(),
    )


def _training_grouping(
    grouping: str | Path | Mapping[str, Any], run_dirs: Sequence[str | Path] | None
) -> Record:
    """Check a declared training comparison and resolve its exact input paths.

    grouping is a schema-1 or schema-2 mapping or JSON/YAML file. Schema 2 adds
    a measure, complete cell weights and declared recipe choices. Python calls
    need no YAML dependency; YAML files require the optional hydra extra.
    File-relative run paths resolve
    from that file; mapping paths resolve from the current directory. Require
    explicit cells, panel, depth, groups and independent-run units. Available
    run_dirs must match the declaration exactly; None uses all declared paths.
    Missing declared runs remain
    reportable. Return a copied declaration with absolute paths. Invalid fields,
    duplicate paths, unknown groups or undeclared inputs raise ValueError.
    """
    if isinstance(grouping, (str, Path)):
        source = Path(grouping).expanduser().resolve()
        base = source.parent
        raw = read_declaration(source)
    else:
        base = Path.cwd()
        raw = dict(grouping)
    required = {
        "schema_version",
        "maps",
        "opponents",
        "panel_digest",
        "red_zone_depth",
        "runs",
    }
    allowed = required | {"comparisons", "bootstrap_draws", "bootstrap_seed"}
    if raw.get("schema_version") == 2:
        allowed |= {"measure", "cell_weights", "recipe_selection"}
    if not required <= raw.keys() or raw.keys() - allowed:
        raise ValueError("Grouping needs the declared fields for schema 1 or 2 only")
    if type(raw["schema_version"]) is not int or raw["schema_version"] not in (1, 2):
        raise ValueError("Grouping schema_version must be 1 or 2")
    maps, opponents = raw["maps"], raw["opponents"]
    if not isinstance(maps, list) or not maps:
        raise ValueError("Grouping maps must be a nonempty list")
    maps = [_integer(value, "grouping map") for value in cast(list[Any], maps)]
    if len(set(maps)) != len(maps):
        raise ValueError("Grouping maps must be distinct")
    if (
        not isinstance(opponents, list)
        or not opponents
        or any(
            not isinstance(name, str) or not name for name in cast(list[Any], opponents)
        )
        or len(set(cast(list[Any], opponents))) != len(cast(list[Any], opponents))
    ):
        raise ValueError("Grouping opponents must be distinct nonempty names")
    opponents = cast(list[str], opponents)
    if not isinstance(raw["panel_digest"], str) or not raw["panel_digest"]:
        raise ValueError("Grouping needs the saved panel_digest")
    _red_zone_label(raw["red_zone_depth"])
    draws = _integer(raw.get("bootstrap_draws", 2000), "bootstrap_draws", minimum=1)
    seed = _integer(raw.get("bootstrap_seed", 19_044_003), "bootstrap_seed")
    if not isinstance(raw["runs"], list) or not raw["runs"]:
        raise ValueError("Grouping runs must declare every planned run")
    runs: list[Record] = []
    for raw_row in cast(list[Any], raw["runs"]):
        if not isinstance(raw_row, dict):
            raise ValueError("Each grouped run must be an object")
        row = cast(Record, raw_row)
        if set(row) != {"run_dir", "group", "unit"}:
            raise ValueError("Each grouped run needs run_dir, group and unit")
        if any(not isinstance(row[key], str) or not row[key] for key in row):
            raise ValueError(
                "Grouped run paths, groups and units must be nonempty text"
            )
        runs.append(
            {
                **row,
                "run_dir": str((base / Path(row["run_dir"]).expanduser()).resolve()),
            }
        )
    declared = [row["run_dir"] for row in runs]
    supplied = (
        declared
        if run_dirs is None
        else [str(Path(path).expanduser().resolve()) for path in run_dirs]
    )
    if len(set(declared)) != len(declared) or len(set(supplied)) != len(supplied):
        raise ValueError("Grouping and analysis cannot repeat a run directory")
    if set(supplied) - set(declared) or any(
        Path(path).exists() and path not in supplied for path in declared
    ):
        raise ValueError(
            "Available analysis inputs must match the declared grouped runs"
        )
    names = {row["group"] for row in runs}
    comparisons = raw.get("comparisons", [])
    if not isinstance(comparisons, list):
        raise ValueError("Grouping comparisons must be a list")
    comparisons = cast(list[Any], comparisons)
    seen: set[tuple[str, str, bool]] = set()
    for raw_row in comparisons:
        if not isinstance(raw_row, dict):
            raise ValueError("Each comparison must be an object")
        row = cast(Record, raw_row)
        if set(row) != {"left", "right", "paired"}:
            raise ValueError("Each comparison needs left, right and paired")
        if not isinstance(row["left"], str) or not isinstance(row["right"], str):
            raise ValueError("Comparison groups must be names")
        if (
            row["left"] not in names
            or row["right"] not in names
            or row["left"] == row["right"]
        ):
            raise ValueError("Comparisons must name two distinct declared groups")
        if type(row["paired"]) is not bool:
            raise ValueError("Comparison paired must be true or false")
        identity = row["left"], row["right"], row["paired"]
        if identity in seen:
            raise ValueError("Grouping cannot repeat a comparison")
        seen.add(identity)
    controls = (
        {}
        if raw["schema_version"] == 1
        else _grouping_controls(raw, maps, opponents, names)
    )
    return {
        **raw,
        **controls,
        "maps": maps,
        "opponents": list(opponents),
        "runs": runs,
        "comparisons": [dict(row) for row in comparisons],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def _grouping_controls(
    raw: Mapping[str, Any],
    maps: Sequence[int],
    opponents: Sequence[str],
    groups: set[str],
) -> Record:
    """Check schema-2 measure, complete cell weights and named recipe comparisons.

    raw is the declared mapping; maps/opponents fix cell order and groups names
    the declared recipes. Return copied choices with original and normalized
    weights. Every cell, including zero-weight cells, remains required evidence.
    Weights are finite nonnegative numbers with positive total. Recipe choices
    need distinct known groups, maximize/minimize and an exact tie permutation.
    A choice may also declare a base rule with all three of base (one of its
    groups), min_margin_gain (a finite number of at least zero, in the measure's
    units) and min_units_won (an integer of at least zero). A partial base rule,
    a base outside the choice or invalid numbers raise ValueError. The returned
    min_margin_gain is a float. No result is inspected and no files change.
    """
    measure = raw.get("measure", "game_score")
    if measure not in ("game_score", "point_margin"):
        raise ValueError("Grouping measure must be game_score or point_margin")
    cells = [(map_id, name) for map_id in maps for name in opponents]
    declared = raw.get("cell_weights")
    if declared is None:
        declared = [
            {"map_id": map_id, "opponent": name, "weight": 1.0}
            for map_id, name in cells
        ]
    if not isinstance(declared, list):
        raise ValueError("cell_weights must list every declared map/opponent cell")
    indexed: dict[tuple[int, str], float] = {}
    for value in cast(list[Any], declared):
        if not isinstance(value, dict) or set(cast(Record, value)) != {
            "map_id",
            "opponent",
            "weight",
        }:
            raise ValueError("Each cell weight needs map_id, opponent and weight")
        value = cast(Record, value)
        key = (_integer(value["map_id"], "weight map"), value["opponent"])
        weight = value["weight"]
        if (
            not isinstance(key[1], str)
            or key not in cells
            or key in indexed
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Cell weights need distinct cells and finite nonnegative weights"
            )
        indexed[key] = float(weight)
    if set(indexed) != set(cells) or not indexed or max(indexed.values()) <= 0:
        raise ValueError(
            "Cell weights must cover every declared cell with positive total"
        )
    scale = max(indexed.values())
    total = sum(value / scale for value in indexed.values())
    weights = [
        {
            "map_id": map_id,
            "opponent": name,
            "weight": indexed[(map_id, name)],
            "normalized_weight": (indexed[(map_id, name)] / scale) / total,
        }
        for map_id, name in cells
    ]
    selections = raw.get("recipe_selection", [])
    if not isinstance(selections, list):
        raise ValueError("recipe_selection must be a list of named choices")
    choices: list[Record] = []
    names: set[str] = set()
    base_rule = {"base", "min_margin_gain", "min_units_won"}
    for value in cast(list[Any], selections):
        if not isinstance(value, dict) or set(cast(Record, value)) - base_rule != {
            "name",
            "groups",
            "direction",
            "tie_order",
        }:
            raise ValueError(
                "Recipe choice needs name, groups, direction and tie_order"
            )
        value = cast(Record, value)
        if set(value) & base_rule and not base_rule <= set(value):
            raise ValueError(
                "A base rule needs base, min_margin_gain and min_units_won together"
            )
        name, selected, order = value["name"], value["groups"], value["tie_order"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Recipe choice names must be distinct and nonempty")
        if (
            not isinstance(selected, list)
            or not selected
            or any(
                not isinstance(group, str) or group not in groups
                for group in cast(list[Any], selected)
            )
            or len(set(cast(list[str], selected))) != len(cast(list[str], selected))
        ):
            raise ValueError("Recipe choices need distinct existing groups")
        if (
            not isinstance(order, list)
            or any(not isinstance(group, str) for group in cast(list[Any], order))
            or len(cast(list[str], order)) != len(cast(list[str], selected))
            or set(cast(list[str], order)) != set(cast(list[str], selected))
        ):
            raise ValueError("Recipe tie_order must be a complete group permutation")
        if value["direction"] not in ("maximize", "minimize"):
            raise ValueError("Recipe direction must be maximize or minimize")
        rule: Record = {}
        if "base" in value:
            members, gain = cast(list[str], selected), value["min_margin_gain"]
            if value["base"] not in members or len(members) < 2:
                raise ValueError("A recipe base must be one of two or more groups")
            if (
                isinstance(gain, bool)
                or not isinstance(gain, (int, float))
                or not math.isfinite(gain)
                or gain < 0
            ):
                raise ValueError(
                    "min_margin_gain must be a finite number of at least 0"
                )
            rule = {
                "min_margin_gain": float(gain),
                "min_units_won": _integer(value["min_units_won"], "min_units_won"),
            }
        names.add(name)
        choices.append(
            {
                **value,
                **rule,
                "groups": list(cast(list[str], selected)),
                "tie_order": list(cast(list[str], order)),
            }
        )
    return {
        "measure": measure,
        "cell_weights": [
            {key: value for key, value in row.items() if key != "normalized_weight"}
            for row in weights
        ],
        "normalized_cell_weights": [row["normalized_weight"] for row in weights],
        "recipe_selection": choices,
    }


def _training_lineage(details: Mapping[str, Any]) -> str | None:
    """Read a saved independent training root; unknown continuations stay unknown.

    A valid training_lineage.root_run_id takes precedence. Historical fresh runs
    use their own saved run_id. Parent/continuation markers without that saved
    lineage prevent an independence claim. This reads metadata only.
    """
    lineage = details.get("training_lineage")
    if lineage is not None:
        root = (
            cast(Record, lineage).get("root_run_id")
            if isinstance(lineage, dict)
            else None
        )
        return root if isinstance(root, str) and root else None
    if any(
        details.get(key) is not None
        for key in (
            "continuation",
            "parent_run",
            "parent_run_id",
            "parent_checkpoint",
            "extended_from",
        )
    ):
        return None
    run_id = details.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def _training_run_vector(
    entry: Mapping[str, Any],
    declaration: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    details: Mapping[str, Any],
    point_cells: Mapping[str, Record] | None = None,
) -> Record:
    """Keep one declared run and every selected map/opponent cell in the report.

    Missing or failed runs, unknown lineage and incomplete selected evidence get
    explicit reasons and cannot support a group interval. No final checkpoint is
    substituted. Cell scores remain descriptive evidence even when unavailable
    for aggregation; invalid or absent cell values stay None. No files change.
    """
    selected = summary.get("selection") if summary is not None else None
    selected = cast(Record, selected) if isinstance(selected, dict) else {}
    status = summary.get("status", "unknown") if summary is not None else "missing"
    root = _training_lineage(details)
    reasons = []
    if status != "complete":
        reasons.append(f"Run status is {status}")
    if root is None and summary is not None:
        reasons.append("Independent training lineage is unknown")
    if not selected or selected.get("complete") is not True:
        reasons.append("A complete selected-checkpoint result is unavailable")
    if selected.get("purpose") != "confirmation":
        reasons.append("Selected evidence is not a confirmation result")
    if not isinstance(selected.get("checkpoint_id"), str) or not selected.get(
        "checkpoint_id"
    ):
        reasons.append("Selected checkpoint identity is unavailable")
    if selected.get("panel_digest") != declaration["panel_digest"]:
        reasons.append("Selected panel does not match the declaration")
    if selected.get("red_zone_depth") != declaration["red_zone_depth"]:
        reasons.append("Selected Red Zone depth does not match the declaration")
    if selected.get("maps", declaration["maps"]) != declaration["maps"]:
        reasons.append("Selected maps do not match the declaration")
    expected = [
        (map_id, name)
        for map_id in declaration["maps"]
        for name in declaration["opponents"]
    ]
    expected_set = set(expected)
    raw_cells = selected.get("cells", [])
    indexed: dict[tuple[int, str], Record] = {}
    malformed = not isinstance(raw_cells, list)
    if isinstance(raw_cells, list):
        for raw_cell in cast(list[Any], raw_cells):
            if not isinstance(raw_cell, dict):
                malformed = True
                continue
            cell = cast(Record, raw_cell)
            map_id, opponent = cell.get("map_id"), cell.get("opponent")
            if type(map_id) is not int or not isinstance(opponent, str):
                malformed = True
                continue
            key = map_id, opponent
            if key not in expected_set or key in indexed:
                malformed = True
            indexed[key] = cell
    recovered: Record = (point_cells or {}).get(str(selected.get("task_id")), {})
    if recovered:
        expected_record = json.loads(json.dumps(recovered["summary"]))
        for actual, expected_cell in zip(
            selected.get("cells", []), expected_record["cells"], strict=False
        ):
            for field in ("mean_team_a_score", "mean_team_b_score"):
                if field not in actual:
                    expected_cell.pop(field, None)
        additions = {
            "elapsed_seconds",
            "routine_comparison",
            "confirmation_comparison",
            "artifact_id",
            "actor_path",
            "result_source",
        }
        # Historical pre-Red-Zone verification can add this derived field only
        # to the returned record. It does not change the original summary.
        if "mean_kill_difference" not in expected_record:
            additions.add("mean_kill_difference")
        if {
            key: value for key, value in selected.items() if key not in additions
        } != expected_record:
            raise ValueError(
                "Selected record differs from its verified native-point evidence"
            )
        if (
            "result_source" in selected
            and selected["result_source"] != recovered["result_source"]
        ):
            raise ValueError(
                "Selected native-point source differs from checked evidence"
            )
    recovered_cells = {
        (cell["map_id"], cell["opponent"]): cell for cell in recovered.get("cells", [])
    }
    cells: list[Record] = []
    for map_id, opponent in expected:
        raw = {
            **recovered_cells.get((map_id, opponent), {}),
            **indexed.get((map_id, opponent), {}),
        }
        score = raw.get("score")
        score = (
            score
            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(score)
                and 0 <= score <= 1
            )
            else None
        )
        points = tuple(
            value
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
            else None
            for value in (raw.get("mean_team_a_score"), raw.get("mean_team_b_score"))
        )
        margin = (
            (float(cast(float, points[0])) - float(cast(float, points[1])))
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in points
            )
            else None
        )
        if margin is not None and not math.isfinite(margin):
            margin = None
        value = margin if declaration.get("measure") == "point_margin" else score
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (
                declaration.get("measure", "game_score") == "game_score"
                and not 0 <= value <= 1
            )
        ):
            value = None
            malformed = True
        cells.append(
            {
                "map_id": map_id,
                "opponent": opponent,
                "score": value,
                **(
                    {
                        "game_score": score,
                        "point_margin": margin,
                        "mean_team_a_score": points[0],
                        "mean_team_b_score": points[1],
                    }
                    if declaration["schema_version"] == 2
                    else {}
                ),
            }
        )
    if malformed or set(indexed) != expected_set:
        reasons.append("Selected results lack the complete declared map/opponent cells")
    config = details.get("config", {})
    seed = cast(Record, config).get("seed") if isinstance(config, dict) else None
    return {
        **entry,
        "run_id": summary.get("run_id") if summary else None,
        "training_seed": seed if type(seed) is int and seed >= 0 else None,
        "status": status,
        "lineage_root_run_id": root,
        "checkpoint_id": selected.get("checkpoint_id"),
        "actor_digest": selected.get("actor_digest"),
        **(
            {
                "task_id": selected.get("task_id"),
                "result_source": selected.get(
                    "result_source", recovered.get("result_source")
                ),
                "pass_paths": selected.get("pass_paths", []),
            }
            if declaration["schema_version"] == 2
            else {}
        ),
        "cells": cells,
        "available": not reasons,
        "unavailable_reasons": reasons,
    }


def _training_vector_estimate(
    left: np.ndarray[Any, np.dtype[np.float64]],
    *,
    draws: int,
    seed: int,
    right: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    weights: Sequence[float] | None = None,
) -> Record:
    """Resample independent rows intact, retaining their cross-cell dependence.

    left has one full condition vector per independent run or matched run pair.
    Optional right contains an independent comparison group; its sampled mean
    is subtracted from left. Rows in separate arrays are never treated as paired.
    weights optionally gives one checked normalized weight per cell; None
    preserves the historical equal-cell arithmetic. Return the weighted mean
    and 95% percentile bounds, plus each cell's descriptive mean and bounds.
    One-row groups retain means but no interval.
    Draw indices are reused across all cells and use a private NumPy generator.
    Memory grows with draws times runs, not draws times runs times cells.
    """
    valid_interval = len(left) >= 2 and (right is None or len(right) >= 2)
    random = np.random.default_rng(seed)
    left_indices = (
        random.integers(len(left), size=(draws, len(left))) if valid_interval else None
    )
    right_indices = (
        random.integers(len(right), size=(draws, len(right)))
        if valid_interval and right is not None
        else None
    )
    sampled_means = np.zeros(draws, np.float64)
    cells: list[Record] = []
    for index in range(left.shape[1]):
        score = float(left[:, index].mean())
        if right is not None:
            score -= float(right[:, index].mean())
        low, high = None, None
        if left_indices is not None:
            samples = left[:, index][left_indices].mean(axis=1)
            if right_indices is not None and right is not None:
                samples -= right[:, index][right_indices].mean(axis=1)
            sampled_means += (
                samples / left.shape[1] if weights is None else samples * weights[index]
            )
            bounds = np.percentile(samples, (2.5, 97.5))
            low, high = float(bounds[0]), float(bounds[1])
        cells.append({"score": score, "ci_low": low, "ci_high": high})
    bounds = np.percentile(sampled_means, (2.5, 97.5)) if valid_interval else None
    return {
        "score": float(np.mean([row["score"] for row in cells]))
        if weights is None
        else float(
            sum(
                row["score"] * weight
                for row, weight in zip(cells, weights, strict=True)
            )
        ),
        "ci_low": float(bounds[0]) if bounds is not None else None,
        "ci_high": float(bounds[1]) if bounds is not None else None,
        "cells": cells,
        "unavailable_reasons": []
        if valid_interval
        else ["Fewer than two independent training units"],
    }


def _training_variation(
    declaration: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    details_by_path: Mapping[str, Mapping[str, Any]],
    point_cells: Mapping[str, Record] | None = None,
) -> Record:
    """Report declared training groups from saved per-run selected records.

    declaration is checked by _training_grouping. summaries provide each run's
    selection, status and identity; explicit re-selection may replace selection
    before this call. details_by_path provides saved lineage and selection rules.
    A checked re-selection declaration replaces each original rule. Missing runs,
    cells or pairs invalidate the full declared comparison rather than shrinking
    it. independent_units counts only supported independent runs; unit, seed or
    lineage conflicts leave that count unknown. Shared saved seeds and lineage
    across groups require pairing; each shared identity must stay within one
    declared pair. Missing cell results alone do not change a known independent
    count. Return JSON-safe group/cell estimates, exact run identities, counts
    and reasons. Fixed-actor game intervals are a separate
    report, never resampled as training runs. No source file, learner state or
    global random generator changes.
    """
    summaries_by_path = {str(row["run_dir"]): row for row in summaries}
    runs = [
        _training_run_vector(
            entry,
            declaration,
            summaries_by_path.get(entry["run_dir"]),
            details_by_path.get(entry["run_dir"], {}),
            point_cells,
        )
        for entry in declaration["runs"]
    ]
    from marl_battlegrounds.training.validation import validation_conditions

    conditions = [
        validation_conditions(
            {
                **summary["selection"],
                "selection_rule": (
                    summary["selection_decision"]["rule"]
                    if summary.get("selection_decision") is not None
                    else details_by_path.get(run["run_dir"], {}).get(
                        "selection_rule", "saved"
                    )
                ),
            }
        )
        for run in runs
        if run["available"]
        for summary in (summaries_by_path[run["run_dir"]],)
        if isinstance(summary.get("selection"), Mapping)
    ]
    if conditions and any(row != conditions[0] for row in conditions[1:]):
        for run in runs:
            run["available"] = False
            run["unavailable_reasons"].append(
                "Selected validation conditions differ across declared runs"
            )
    weights = (
        None
        if declaration["schema_version"] == 1
        else declaration["normalized_cell_weights"]
    )
    cell_keys = [
        {"map_id": map_id, "opponent": name}
        for map_id in declaration["maps"]
        for name in declaration["opponents"]
    ]
    common: Record = {
        "confidence": 0.95,
        "uncertainty": "Training variation; fixed validation field and selection rule",
        "bootstrap_draws": declaration["bootstrap_draws"],
        "bootstrap_seed": declaration["bootstrap_seed"],
    }
    groups: list[Record] = []
    members: dict[str, list[Record]] = {}
    arrays: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    for name in dict.fromkeys(row["group"] for row in runs):
        rows = [row for row in runs if row["group"] == name]
        members[name] = rows
        roots = [row["lineage_root_run_id"] for row in rows]
        units = [row["unit"] for row in rows]
        reasons = []
        if not all(row["available"] for row in rows):
            reasons.append(
                "Not every declared run has complete compatible selected evidence"
            )
        independence_reasons = []
        if len(set(units)) != len(units):
            independence_reasons.append(
                "Declared runs repeat an independent training unit"
            )
        saved_seeds = [
            row["training_seed"] for row in rows if row["training_seed"] is not None
        ]
        if len(set(saved_seeds)) != len(saved_seeds):
            independence_reasons.append(
                "Saved runs repeat the same training seed within this group"
            )
        if len({root for root in roots if root is not None}) != len(roots):
            independence_reasons.append(
                "Saved training lineage is unknown or repeats within this group"
            )
        reasons.extend(independence_reasons)
        estimate: Record = {
            "score": None,
            "ci_low": None,
            "ci_high": None,
            "cells": [
                {"score": None, "ci_low": None, "ci_high": None} for _ in cell_keys
            ],
            "unavailable_reasons": reasons,
        }
        if not reasons:
            values = np.asarray(
                [[cell["score"] for cell in row["cells"]] for row in rows], np.float64
            )
            arrays[name] = values
            estimate = _training_vector_estimate(
                values,
                draws=common["bootstrap_draws"],
                seed=common["bootstrap_seed"],
                weights=weights,
            )
        groups.append(
            {
                **common,
                "group": name,
                "sampling_unit": "Whole independent training run",
                "method": "Percentile bootstrap of whole run vectors",
                "declared_runs": len(rows),
                "complete_runs": sum(row["status"] == "complete" for row in rows),
                "available_runs": sum(row["available"] for row in rows),
                "independent_units": len(rows) if not independence_reasons else None,
                **estimate,
                "cells": [
                    {**key, **value}
                    for key, value in zip(cell_keys, estimate["cells"], strict=True)
                ],
            }
        )
    comparisons = []
    for comparison in declaration["comparisons"]:
        left, right = comparison["left"], comparison["right"]
        paired = comparison["paired"]
        left_rows, right_rows = members[left], members[right]
        left_units, right_units = (
            [row["unit"] for row in left_rows],
            [row["unit"] for row in right_rows],
        )
        reasons = []
        if left not in arrays or right not in arrays:
            reasons.append(
                "Both full declared groups need complete independent-run evidence"
            )
        if paired and set(left_units) != set(right_units):
            reasons.append(
                "Paired groups must contain the same declared training units"
            )
        left_roots = {row["lineage_root_run_id"] for row in left_rows}
        right_roots = {row["lineage_root_run_id"] for row in right_rows}
        root_units: dict[str, set[str]] = {}
        seed_units: dict[int, set[str]] = {}
        for row in (*left_rows, *right_rows):
            root = row["lineage_root_run_id"]
            if root is not None:
                root_units.setdefault(root, set()).add(row["unit"])
            saved_seed = row["training_seed"]
            if saved_seed is not None:
                seed_units.setdefault(saved_seed, set()).add(row["unit"])
        if paired and any(len(units) > 1 for units in root_units.values()):
            reasons.append("Saved training lineage crosses declared run pairs")
        if paired and any(len(units) > 1 for units in seed_units.values()):
            reasons.append("Saved training seeds cross declared run pairs")
        left_seeds = {row["training_seed"] for row in left_rows}
        right_seeds = {row["training_seed"] for row in right_rows}
        if not paired and (left_seeds & right_seeds) - {None}:
            reasons.append(
                "Shared saved training seeds require a declared paired comparison"
            )
        if not paired and (
            set(left_units) & set(right_units) or (left_roots & right_roots) - {None}
        ):
            reasons.append(
                "Related training units require a declared paired comparison"
            )
        estimate = {
            "score": None,
            "ci_low": None,
            "ci_high": None,
            "cells": [
                {"score": None, "ci_low": None, "ci_high": None} for _ in cell_keys
            ],
            "unavailable_reasons": reasons,
        }
        if not reasons:
            left_values, right_values = arrays[left], arrays[right]
            if paired:
                order = [right_units.index(unit) for unit in left_units]
                estimate = _training_vector_estimate(
                    left_values - right_values[order],
                    draws=common["bootstrap_draws"],
                    seed=common["bootstrap_seed"],
                    weights=weights,
                )
            else:
                estimate = _training_vector_estimate(
                    left_values,
                    right=right_values,
                    draws=common["bootstrap_draws"],
                    seed=common["bootstrap_seed"],
                    weights=weights,
                )
        comparisons.append(
            {
                **common,
                **comparison,
                "score_direction": "Left minus right",
                "sampling_unit": "Whole matched training-run pair"
                if paired
                else "Whole independent training run in each group",
                "method": "Percentile bootstrap of matched run differences"
                if paired
                else "Independent percentile bootstrap of whole run vectors",
                "left_declared_runs": len(left_rows),
                "right_declared_runs": len(right_rows),
                "paired_units": len(left_units)
                if paired
                and set(left_units) == set(right_units)
                and len(set(left_units)) == len(left_units)
                else None,
                **estimate,
                "cells": [
                    {**key, **value}
                    for key, value in zip(cell_keys, estimate["cells"], strict=True)
                ],
            }
        )
    return {
        "schema_version": declaration["schema_version"],
        "declaration": {
            key: value
            for key, value in declaration.items()
            if key != "normalized_cell_weights"
        },
        "runs": runs,
        "groups": groups,
        "comparisons": comparisons,
        **(
            {"recipe_selection": _recipe_decisions(declaration, runs, groups)}
            if declaration["schema_version"] == 2
            else {}
        ),
    }


def _recipe_decisions(
    declaration: Mapping[str, Any], runs: Sequence[Record], groups: Sequence[Record]
) -> Record:
    """Choose declared recipes by equally weighted independent-run means.

    Parameters
    ----------
    declaration : mapping
        The grouping checked by _training_grouping: measure, cells, normalized
        weights and each choice's groups, direction and tie order. A choice may
        add a base rule (base, min_margin_gain, min_units_won).
    runs : sequence of dict
        One record per declared run from _training_run_vector, in declared order.
    groups : sequence of dict
        One estimate per group from the existing vector reducer.

    Returns
    -------
    dict
        Deterministic JSON-safe record with every mean, reason, weight and
        source reference, plus declaration_id and content_id hashes. Each
        decision's winner is None until every candidate group and unit is
        available with a known training seed and lineage.

    Notes
    -----
    Without a base rule the winner is the best mean in the declared direction;
    tie_order breaks exact ties. One complete unit permits a labelled software
    choice, never an interval.

    With a base rule every other group is a challenger tested against the base.
    A challenger passes only when both hold: its mean beats the base mean by
    more than min_margin_gain (exactly equal does not pass), and it beats the
    base on at least min_units_won paired units. "Beats" follows the declared
    direction. A unit's value is that run's weighted cell measure, and units
    pair by their declared unit name. Each decision then adds "challengers"
    (one record per challenger with mean_gain, units_won, paired_units, per-unit
    values and passed) and "passed_challengers" (passing groups in tie order).
    The winner is the base when no challenger passes; otherwise it is the
    passing challenger with the best mean, then tie order. To combine every
    passing change, read passed_challengers. With incomplete evidence, passed
    and the winner stay None. Choices without a base rule keep their earlier
    record shape exactly. No checkpoint, run or file changes.
    """
    estimates = {row["group"]: row for row in groups}
    weights = declaration["normalized_cell_weights"]

    def unit_value(row: Record) -> float | None:
        """Return one available run's weighted cell measure, or None."""
        if not row["available"]:
            return None
        return math.fsum(
            cell["score"] * weight
            for cell, weight in zip(row["cells"], weights, strict=True)
        )

    decisions: list[Record] = []
    for choice in declaration["recipe_selection"]:
        candidates: list[Record] = []
        for name in choice["groups"]:
            members = [row for row in runs if row["group"] == name]
            estimate = estimates[name]
            reasons = list(
                dict.fromkeys(
                    reason for row in members for reason in row["unavailable_reasons"]
                )
            )
            if any(row["training_seed"] is None for row in members):
                reasons.append("A declared training seed is unknown")
            if estimate["independent_units"] != len(members):
                reasons.append("Training units are not known independent families")
            if estimate["score"] is None:
                reasons.append("A complete compatible recipe mean is unavailable")
            candidates.append(
                {
                    "group": name,
                    "mean": estimate["score"],
                    "available_mean": (
                        math.fsum(
                            math.fsum(
                                cell["score"] * weight
                                for cell, weight in zip(
                                    row["cells"],
                                    declaration["normalized_cell_weights"],
                                    strict=True,
                                )
                            )
                            / sum(item["available"] for item in members)
                            for row in members
                            if row["available"]
                        )
                        if any(row["available"] for row in members)
                        else None
                    ),
                    "available_units": sum(row["available"] for row in members),
                    "declared_units": len(members),
                    "eligible": not reasons,
                    "unavailable_reasons": reasons,
                    "sources": [dict(row) for row in members],
                }
            )
        complete = all(row["eligible"] for row in candidates)
        order = {name: index for index, name in enumerate(choice["tie_order"])}
        sign = 1.0 if choice["direction"] == "maximize" else -1.0
        contenders = candidates
        base_record: Record = {}
        if "base" in choice:
            base = choice["base"]
            means = {row["group"]: row["mean"] for row in candidates}
            base_units = {
                row["unit"]: unit_value(row) for row in runs if row["group"] == base
            }
            challengers: list[Record] = []
            for name in choice["tie_order"]:
                if name == base:
                    continue
                units = {
                    row["unit"]: unit_value(row) for row in runs if row["group"] == name
                }
                reasons = []
                if not complete:
                    reasons.append("Every group in this choice needs complete evidence")
                if set(units) != set(base_units):
                    reasons.append("Challenger and base must declare the same units")
                unit_rows: list[Record] = []
                for unit in sorted(set(units) | set(base_units)):
                    mine, theirs = units.get(unit), base_units.get(unit)
                    unit_rows.append(
                        {
                            "unit": unit,
                            "challenger": mine,
                            "base": theirs,
                            "won": None
                            if reasons or mine is None or theirs is None
                            else sign * (mine - theirs) > 0,
                        }
                    )
                gain = None if reasons else sign * (means[name] - means[base])
                won = None if reasons else sum(row["won"] is True for row in unit_rows)
                challengers.append(
                    {
                        "group": name,
                        "base": base,
                        "mean_gain": gain,
                        "min_margin_gain": choice["min_margin_gain"],
                        "units_won": won,
                        "paired_units": None if reasons else len(unit_rows),
                        "min_units_won": choice["min_units_won"],
                        "units": unit_rows,
                        "passed": None
                        if gain is None or won is None
                        else gain > choice["min_margin_gain"]
                        and won >= choice["min_units_won"],
                        "unavailable_reasons": reasons,
                    }
                )
            passed = [row["group"] for row in challengers if row["passed"]]
            contenders = [row for row in candidates if row["group"] in passed] or [
                row for row in candidates if row["group"] == base
            ]
            base_record = {
                "challengers": challengers,
                "passed_challengers": passed if complete else None,
            }
        winner = (
            min(
                contenders,
                key=lambda row: (-sign * row["mean"], order[row["group"]]),
            )["group"]
            if complete
            else None
        )
        decisions.append(
            {
                **choice,
                **base_record,
                "status": "complete" if complete else "incomplete",
                "winner": winner,
                "candidates": candidates,
                "qualification": (
                    "Software choice; one unit does not support "
                    "training-seed uncertainty"
                ),
            }
        )
    content = {
        "schema_version": 1,
        "declaration_id": hashlib.sha256(
            json.dumps(dict(declaration), sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "measure": declaration["measure"],
        "units": "Native Team A minus Team B points per game"
        if declaration["measure"] == "point_margin"
        else "Native game score (0 to 1)",
        "cell_weights": declaration["cell_weights"],
        "normalized_cell_weights": declaration["normalized_cell_weights"],
        "decisions": decisions,
    }
    return {
        **content,
        "content_id": hashlib.sha256(
            json.dumps(content, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }


def _write_training_variation(
    report: Mapping[str, Any], destination: Path
) -> dict[str, str]:
    """Write declared training variation beside the ordinary read-only report.

    Return JSON/CSV paths and append a plain Markdown summary to run_summary.md.
    Every run remains visible with identity, status and unavailable reasons. CSV
    rows distinguish group means from left-minus-right comparisons and retain
    each declared cell. Only derived report files change, using atomic writes.
    """
    json_path, csv_path = (
        destination / "training_variation.json",
        destination / "training_variation.csv",
    )
    _atomic_text(
        json_path, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    rows: list[Record] = []
    lines = [
        "",
        "## Declared Training Variation",
        "",
        "These 95% intervals resample whole independent training runs, keeping "
        "their map/opponent cells together. They do not replace fixed-actor "
        "game-sampling intervals. Missing runs or cells never become losses "
        "or a smaller comparison.",
        "",
        "### Runs",
        "",
    ]
    for row in report["runs"]:
        lines.append(
            f"- {row['group']} / {row['unit']}: {row['status']}; "
            f"checkpoint `{row['checkpoint_id']}`; "
            + (
                "Complete selected evidence."
                if row["available"]
                else "; ".join(row["unavailable_reasons"]) + "."
            )
        )
    for kind in ("groups", "comparisons"):
        for estimate in report[kind]:
            label = estimate.get(
                "group", f"{estimate.get('left')} minus {estimate.get('right')}"
            )
            reasons = "; ".join(estimate["unavailable_reasons"])
            lines.extend(
                [
                    "",
                    f"### {label}",
                    "",
                    f"Sampling Unit: {estimate['sampling_unit']}. "
                    f"Method: {estimate['method']}.",
                    f"Score: {estimate['score']}. 95% Interval: "
                    + (
                        f"{estimate['ci_low']} to {estimate['ci_high']}."
                        if not reasons
                        else f"Unavailable. {reasons}."
                    ),
                    "",
                ]
            )
            for cell in estimate["cells"]:
                rows.append(
                    {
                        "kind": kind,
                        "name": label,
                        **{
                            key: value
                            for key, value in estimate.items()
                            if not isinstance(value, (dict, list))
                        },
                        **cell,
                        "unavailable_reasons": reasons,
                    }
                )
    _atomic_text(csv_path, _csv(rows))
    summary_path = destination / "run_summary.md"
    _atomic_text(summary_path, summary_path.read_text() + "\n".join(lines) + "\n")
    extra: dict[str, str] = {}
    if report.get("recipe_selection", {}).get("decisions"):
        path = destination / "recipe_selection.json"
        _atomic_text(
            path,
            json.dumps(
                report["recipe_selection"], indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
        )
        extra["recipe_selection"] = str(path)
    return {
        **extra,
        "training_variation": str(json_path),
        "training_variation_cells": str(csv_path),
    }


def _recipe_curve_rows(
    declaration: Mapping[str, Any],
    inputs: Mapping[str, tuple[object, Sequence[Record]]],
) -> list[Record]:
    """Turn saved routine validation into one point-margin row per run and step.

    Parameters
    ----------
    declaration : mapping
        A grouping checked by _training_grouping. Its maps, opponents,
        panel_digest, red_zone_depth, runs and optional normalized cell weights
        decide which results and cells count.
    inputs : mapping of str to (object, sequence of dict)
        Run directory to (saved config method, validation result list). A
        declared run that is missing here, such as one that never started, adds
        no rows.

    Returns
    -------
    list of dict
        Rows in declared run order, then by env_steps, with the columns group,
        unit, run_dir, method, env_steps, point_margin and cells_used.
        point_margin is our points minus the opponent's per game, in native
        points. It is None unless every declared map/opponent cell has finite
        mean_team_a_score and mean_team_b_score. cells_used counts the declared
        cells with finite points.

    Raises
    ------
    ValueError
        A used result has an env_steps value that is not an integer of at least 0.

    Notes
    -----
    Only complete "routine" and "initialization" results (step 0 included)
    with the declared panel_digest and red_zone_depth count. Cells outside the
    declared maps and opponents are ignored. With equal cell weights (the
    default and every schema-1 grouping) the margin comes from
    _mean_point_margin, the owner checkpoint selection uses. Declared unequal
    weights apply the grouping's normalized weights to each cell's margin, as
    recipe selection does. If a run saved two results at one step, the later
    one in the file wins. No file changes.
    """
    cells = [
        (map_id, name)
        for map_id in declaration["maps"]
        for name in declaration["opponents"]
    ]
    weights = declaration.get("normalized_cell_weights") or [1 / len(cells)] * len(
        cells
    )
    rows: list[Record] = []
    for entry in declaration["runs"]:
        method, results = inputs.get(entry["run_dir"], (None, []))
        by_step: dict[int, Record] = {}
        for result in results:
            if (
                result.get("purpose") not in ("routine", "initialization")
                or result.get("complete") is not True
                or result.get("panel_digest") != declaration["panel_digest"]
                or result.get("red_zone_depth") != declaration["red_zone_depth"]
            ):
                continue
            indexed: dict[tuple[object, object], Record] = {}
            for cell in cast(list[Any], result.get("cells") or []):
                if isinstance(cell, dict):
                    cell = cast(Record, cell)
                    indexed[(cell.get("map_id"), cell.get("opponent"))] = cell
            chosen = [
                indexed[key]
                for key in cells
                if key in indexed
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in (
                        indexed[key].get("mean_team_a_score"),
                        indexed[key].get("mean_team_b_score"),
                    )
                )
            ]
            margin = None
            if len(chosen) == len(cells):
                margin = (
                    _mean_point_margin({"cells": chosen})
                    if len(set(weights)) == 1
                    else math.fsum(
                        _mean_point_margin({"cells": [cell]}) * weight
                        for cell, weight in zip(chosen, weights, strict=True)
                    )
                )
            steps = _integer(result.get("env_steps"), "env_steps")
            by_step[steps] = {
                "group": entry["group"],
                "unit": entry["unit"],
                "run_dir": entry["run_dir"],
                "method": method if isinstance(method, str) else "unknown",
                "env_steps": steps,
                "point_margin": margin,
                "cells_used": len(chosen),
            }
        rows.extend(by_step[steps] for steps in sorted(by_step))
    return rows


def _write_recipe_curves(
    declaration: Mapping[str, Any],
    inputs: Mapping[str, tuple[object, Sequence[Record]]],
    destination: Path,
    plt: Any,  # noqa: ANN401
) -> tuple[dict[str, str], str]:
    """Write each recipe's validation point margin over training.

    Parameters
    ----------
    declaration : mapping
        A grouping checked by _training_grouping. Groups are recipes and units
        are seeds.
    inputs : mapping of str to (object, sequence of dict)
        Run directory to (saved config method, validation result list), as read
        by analyze.
    destination : Path
        The report folder.
    plt : module
        The already-imported matplotlib.pyplot module.

    Returns
    -------
    tuple of (dict[str, str], str)
        Paths keyed recipe_curves_png, recipe_curves_svg and recipe_curves_csv,
        and one plain sentence for run_summary.md. When no saved result has a
        point margin, the paths are empty, the sentence says the curves were
        skipped and why, and any old recipe_curves files in destination are
        removed so they cannot be mistaken for this report.

    Notes
    -----
    Rows come from _recipe_curve_rows, so the margin matches selection. The
    figure has one panel per saved method with at least one margin, in declared
    run order; a run without a saved method is labelled unknown. Each seed is
    a thin line; each recipe's mean is a thick line in the same color. The mean
    is drawn only at steps where every seed of that recipe with any margin has
    one, so an unfinished seed shortens the mean instead of changing it. The
    legend names how many seeds each mean uses. The x-axis is environment
    steps. Each file is written to a temporary name and then replaced.
    """
    rows = _recipe_curve_rows(declaration, inputs)
    names = ("png", "svg", "csv")
    if not any(row["point_margin"] is not None for row in rows):
        for suffix in names:
            (destination / f"recipe_curves.{suffix}").unlink(missing_ok=True)
        return {}, (
            "Recipe learning curves were skipped: no saved routine validation "
            "result has points in every declared map and opponent cell."
        )
    methods = list(
        dict.fromkeys(row["method"] for row in rows if row["point_margin"] is not None)
    )
    figure, axes = plt.subplots(
        len(methods),
        1,
        figsize=(11, 4.5 * len(methods)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    groups = list(dict.fromkeys(row["group"] for row in rows))
    for axis, method in zip(axes[:, 0], methods, strict=True):
        for group in groups:
            color = colors[groups.index(group) % len(colors)]
            seeds: dict[str, dict[int, float]] = {}
            for row in rows:
                if (
                    row["method"] == method
                    and row["group"] == group
                    and row["point_margin"] is not None
                ):
                    seeds.setdefault(row["run_dir"], {})[row["env_steps"]] = row[
                        "point_margin"
                    ]
            if not seeds:
                continue
            for values in seeds.values():
                steps = sorted(values)
                axis.plot(
                    steps,
                    [values[step] for step in steps],
                    color=color,
                    linewidth=0.9,
                    alpha=0.4,
                )
            step_sets = [set(values) for values in seeds.values()]
            shared = sorted(step_sets[0].intersection(*step_sets[1:]))
            axis.plot(
                shared,
                [
                    math.fsum(values[step] for values in seeds.values()) / len(seeds)
                    for step in shared
                ],
                color=color,
                linewidth=2.8,
                marker="o" if len(shared) == 1 else None,
                label=f"{group}: Mean Of {len(seeds)} Seeds",
            )
        axis.axhline(0, color="0.6", linewidth=0.8)
        axis.set_title(f"Method: {method}")
        axis.set_xlabel("Environment Steps")
        axis.set_ylabel("Point Margin (Our Points Minus Theirs)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    figure.suptitle(
        "Validation Point Margin By Recipe "
        "(Thin Lines: One Seed Each; Thick Lines: Recipe Mean)"
    )
    paths: dict[str, str] = {}
    for suffix in ("png", "svg"):
        target = destination / f"recipe_curves.{suffix}"
        temporary = destination / f".recipe_curves.{suffix}.tmp"
        figure.savefig(temporary, format=suffix, dpi=150)
        os.replace(temporary, target)
        paths[f"recipe_curves_{suffix}"] = str(target)
    plt.close(figure)
    target = destination / "recipe_curves.csv"
    _atomic_text(target, _csv(rows))
    paths["recipe_curves_csv"] = str(target)
    return paths, (
        "Recipe learning curves: recipe_curves.png, recipe_curves.svg and "
        f"recipe_curves.csv ({len(rows)} rows, one per seed and validation step)."
    )


def _analysis_selection(
    selection: str | Path,
    run_dirs: Sequence[str | Path],
    *,
    point_cells: dict[str, Record] | None = None,
    source_files: list[Record] | None = None,
) -> tuple[Record, dict[str, Record]]:
    """Verify one completed saved decision and its exact requested run identities.

    selection names a decision JSON file. The shared reader verifies current actor
    and game bytes and rebuilds the declared selection without restoring a model.
    Then require the exact run path set and current saved run IDs. Return its
    reference (including the checked selection rule) and per-run winners. Missing
    confirmations, altered evidence or
    mismatched inputs raise before any report is written. Optional point_cells
    receives checked native means during this same read; source_files receives
    the checked path/hash records. No source file changes.
    """
    from marl_battlegrounds.training.selection import read_selection_decision

    path = Path(selection).expanduser().resolve()
    if point_cells is None:
        decision = read_selection_decision(path, verify_evidence=True)
    else:
        from marl_battlegrounds.training.selection import (
            _read_selection_decision,  # pyright: ignore[reportPrivateUsage]
        )

        decision = _read_selection_decision(
            path, verify_evidence=True, point_cells=point_cells
        )
    if source_files is not None:
        source_files.extend(decision["source_files"])
    requested = [str(Path(value).expanduser().resolve()) for value in run_dirs]
    if len(set(requested)) != len(requested) or set(requested) != set(
        decision["run_dirs"]
    ):
        raise ValueError(
            "The selection decision must name exactly the requested run directories"
        )
    runs = {row["run_dir"]: row for row in decision["runs"]}
    for directory in requested:
        details = _read_json(Path(directory) / "run_details.json", None)
        if (
            not isinstance(details, dict)
            or details.get("run_id") != runs[directory]["run_id"]
        ):
            raise ValueError(
                "The selection decision run identity differs from run_details.json"
            )
    return {
        "path": str(path),
        "decision_id": decision["decision_id"],
        "name": decision["name"],
        "rule": decision["declaration"]["rule"],
    }, runs


def _recover_selection_points(
    run_dirs: Sequence[str | Path], *, source_files: list[Record] | None = None
) -> dict[str, Record]:
    """Read missing native points from checked historical selected-game evidence.

    run_dirs are exact run roots. Return task-keyed point cells without changing
    saved selections or loading models. Only omitted point fields may be filled;
    absent source files leave a cell unavailable. Existing corrupt evidence raises.
    Each run is checked once and partial facts are discarded if that check fails.
    Optional source_files receives the same verified file identities for export.
    """
    point_cells: dict[str, Record] = {}
    for value in run_dirs:
        directory = Path(value).expanduser().resolve()
        selected: Record = _read_json(directory / "selection.json", {}) or {}
        cells = selected.get("cells", [])
        if isinstance(selected.get("task_id"), str) and any(
            "mean_team_a_score" not in cell or "mean_team_b_score" not in cell
            for cell in cells
        ):
            from marl_battlegrounds.training._selection_evidence import (
                _panel_path,  # pyright: ignore[reportPrivateUsage]
                read_run_evidence,
            )

            recovered: dict[str, Record] = {}
            try:
                if not (directory / "run_details.json").is_file():
                    continue
                details = _read_json(directory / "run_details.json", None)
                if not isinstance(details, dict) or not isinstance(
                    details.get("config"), dict
                ):
                    raise ValueError(
                        "Saved native-point evidence needs a valid run config"
                    )
                details = cast(Record, details)
                if not _panel_path(directory, details).is_file():
                    continue
                checked = read_run_evidence(directory, point_cells=recovered)
            except FileNotFoundError:
                continue
            point_cells.update(recovered)
            if source_files is not None:
                source_files.extend(checked["source_files"])
    return point_cells


def analyze(
    run_dirs: Sequence[str | Path] | None,
    *,
    output_dir: str | Path,
    grouping: str | Path | Mapping[str, Any] | None = None,
    selection: str | Path | None = None,
) -> Record:
    """Write factual training curves and summaries from saved trainer records.

    Parameters
    ----------
    run_dirs : sequence of paths or None
        None reads all exact run paths from the required grouping declaration.
        Exact run directories containing run_details.json, committed
        training_updates.jsonl and optional validation_results.json/selection.json.
        Validation results are a list, or an object with a results list. Rolled-back
        update logs and validation results must already describe the active attempt.
        If checkpoint_recovery.json exists, active plots/results are withheld until
        recovery finishes; preserved old files are not treated as current results.
    output_dir : path
        Report directory. Only derived CSV, PNG, SVG and run_summary.md are replaced.
    grouping : path or mapping, optional
        An explicit schema-1 or schema-2 declaration of maps, opponents, panel_digest,
        red_zone_depth and runs. Each run names run_dir, group and unit (one
        independent training family). Optional comparisons name left, right and
        paired; paired groups match unit names exactly. bootstrap_draws defaults
        to 2000 and bootstrap_seed to 19044003; confidence is 95%. File-relative
        paths resolve from the declaration; mapping paths resolve from the current
        directory. Available input paths must match the declaration. Missing paths
        remain visible. Only complete selected-checkpoint vectors enter estimates;
        no final actor or missing-cell score is substituted. Whole run vectors stay
        together during resampling. Related continuations do not add independent
        units. Shared saved training seeds or lineage across groups require pairing
        and must stay within the same declared pair. Schema 2 adds measure
        (game_score or native point_margin), optional complete nonnegative
        cell_weights, and named mean-recipe decisions with direction and exact
        tie_order. A decision may add a base rule (base, min_margin_gain,
        min_units_won): a challenger replaces the base only when its mean gain
        is above min_margin_gain and it wins at least min_units_won paired
        units. Zero-weight cells still need evidence. Omitted controls preserve
        equal-cell game scores. Recipe decisions require every declared compatible
        unit and known seed/lineage; unavailable evidence never becomes zero.
        None writes per-run reports
        without adding grouped training-variation estimates.
    selection : path or None, default=None
        A completed re-selection decision JSON file for exactly these run paths
        and saved run IDs. Before writing reports, recheck its source/actor/game
        identities and declared winner through the shared selection reader. This
        hashes artifacts and reads saved games but restores no model; large
        artifacts can make the check take time. Missing confirmations or changed
        evidence raise. Each run summary and grouping uses that decision's winner,
        with original selection records kept separately by reference. The final
        actor remains an extra result. None uses each original selection.json and
        continues to support ordinary unfinished-run reports.

    Returns
    -------
    dict
        Artifact paths, per-run status and a provisional qualification description.
        When any run is QMIX, "qmix_png" and "qmix_svg" add a figure of loss,
        replay rows per game, TD pairs used per transition, exploration
        rate and gradient norms. When any run is PQN-VDN, "pqn_png" and
        "pqn_svg" add a figure of loss, TD pairs used per generated transition,
        the share of used pairs starting on kept rows, exploration rate and
        gradient norms. Runs saved before gradient norms were logged get a
        short note in that panel. Every report includes
        artifacts["exposure"], the path to exposure_history.csv. This table labels
        checkpoint-accepted and uncommitted/superseded marginal rows separately;
        older runs without snapshots have an empty table. Explicit grouping adds
        training_variation.json,
        training_variation.csv and a separately labelled Markdown section, plus a
        training_variation result containing every declared run, cell, method,
        confidence, count and unavailable reason. Explicit selection adds its
        decision path/ID and per-run original-selection references. Schema-2
        recipe decisions add artifacts.recipe_selection, a deterministic JSON
        record of means, eligibility, weights, tie rule and source identities.
        A choice with a base rule also records each challenger's mean gain,
        units won and pass or fail (see _recipe_decisions). Any grouping also
        adds recipe_curves_png, recipe_curves_svg and recipe_curves_csv: each
        recipe's routine validation point margin (step 0 included) for every
        seed, plus a mean line per recipe, one panel per method. The margin
        uses only the grouping's maps and opponents. "recipe_curves" holds one
        sentence, also added to run_summary.md, naming these files or saying
        why they were skipped (no saved result with points in every cell).
        A successful report does not certify useful learning or general competence.
        Summaries retain saved schemas, initial/per-attempt runtime and terminal
        memory snapshots. Missing historical facts stay unknown; no device query
        or new memory measurement occurs.

    Raises
    ------
    ValueError, OSError
        Required identity files, record structures or committed log rows are invalid.
    ImportError
        The viz extra is unavailable. Input evidence is never changed.

    Notes
    -----
        Host-only and headless. Plotting uses original task/shaping units and splits
        different frozen panels and recorded validation Red Zone depths. Training
        and validation depth labels come from their own records; missing historical
        values remain unknown, distinct from recorded zero. CSV rows carry both
        training_red_zone_depth and the row's red_zone_depth. No learner is loaded.
    """
    declared_grouping = (
        _training_grouping(grouping, run_dirs) if grouping is not None else None
    )
    if run_dirs is None and declared_grouping is not None:
        run_dirs = [row["run_dir"] for row in declared_grouping["runs"]]
    if not run_dirs:
        raise ValueError("analyze needs exact run directories or an explicit grouping")
    point_cells: dict[str, Record] | None = (
        {}
        if declared_grouping is not None
        and declared_grouping.get("measure") == "point_margin"
        else None
    )
    selection_reference, selected_runs = (
        _analysis_selection(selection, run_dirs, point_cells=point_cells)
        if selection is not None
        else (None, {})
    )
    if point_cells is not None and selection is None:
        point_cells.update(_recover_selection_points(run_dirs))
    details_by_path: dict[str, Mapping[str, Any]] = {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    curves: list[Record] = []
    exposure_history: list[Record] = []
    cells: list[Record] = []
    summaries: list[Record] = []
    qmix_runs: list[tuple[str, list[Record]]] = []
    pqn_runs: list[tuple[str, list[Record]]] = []
    curve_inputs: dict[str, tuple[object, Sequence[Record]]] = {}
    figure, axes = plt.subplots(
        2, 2, figsize=(14, 10), constrained_layout=True, sharex="col"
    )
    for value in run_dirs:
        directory = Path(value).expanduser().resolve()
        if declared_grouping is not None and _unstarted_run(directory):
            continue
        raw_details = _read_json(directory / "run_details.json", None)
        if not isinstance(raw_details, dict):
            raise ValueError(f"Missing run_details.json in {directory}")
        details = cast(Record, raw_details)
        details_by_path[str(directory)] = details
        run_id = str(details.get("run_id", directory.name))
        config = details.get("config", {})
        treatment = _treatment(config)
        depth = config.get("red_zone_depth")
        run_label = (
            f"{run_id[:8]} / Seed {config.get('seed', '?')} / {treatment} / "
            f"\nTraining Red Zone: {_red_zone_label(depth)}"
        )
        summary = _summary(directory)
        exposure_history.extend(_exposure_history(directory))
        if selection_reference is not None:
            choice = selected_runs[str(directory)]
            summary.update(
                original_selection=summary["selection"],
                original_selection_path=str(directory / "selection.json"),
                original_selected_actor=summary["selected_actor"],
                selection_decision=selection_reference,
                selection=choice["winner"],
                selected_actor=choice["winner"]["actor_path"],
            )
        recovering = summary["status"] == "recovering"
        updates = [] if recovering else _read_rows(directory / "training_updates.jsonl")
        raw_results: object = (
            [] if recovering else _read_json(directory / "validation_results.json", [])
        )
        if isinstance(raw_results, dict):
            raw_results = cast(Record, raw_results).get("results", [])
        if not isinstance(raw_results, list) or any(
            not isinstance(row, dict) for row in cast(list[object], raw_results)
        ):
            raise ValueError("validation_results.json must contain result objects")
        results = cast(list[Record], raw_results)
        curve_inputs[str(directory)] = (config.get("method"), results)
        if config.get("method") == "qmix":
            qmix_runs.append((run_label, updates))
        if config.get("method") == "pqn_vdn":
            pqn_runs.append((run_label, updates))
        for row in updates:
            curves.append(
                {
                    "run_id": run_id,
                    "kind": "update",
                    "training_red_zone_depth": depth,
                    "red_zone_depth": depth,
                    **{
                        key: value
                        for key, value in row.items()
                        if not isinstance(value, (dict, list))
                    },
                }
            )
        for result in results:
            curves.append(
                {
                    "run_id": run_id,
                    "kind": str(result.get("purpose", "validation")),
                    "training_red_zone_depth": depth,
                    "red_zone_depth": result.get("red_zone_depth"),
                    **{
                        key: result.get(key)
                        for key in (
                            "env_steps",
                            "elapsed_seconds",
                            "wall_seconds",
                            "checkpoint_id",
                            "actor_digest",
                            "panel_digest",
                            "complete",
                            "score",
                            "ci_low",
                            "ci_high",
                        )
                    },
                }
            )
            for cell in result.get("cells", []):
                cells.append(
                    {
                        "run_id": run_id,
                        "checkpoint_id": result.get("checkpoint_id"),
                        "purpose": result.get("purpose"),
                        "panel_digest": result.get("panel_digest"),
                        **cell,
                        "training_red_zone_depth": depth,
                        "red_zone_depth": result.get("red_zone_depth"),
                    }
                )
        for x_index, x_name in enumerate(("env_steps", "elapsed_seconds")):
            for name, label in (
                ("task_reward_mean", "Task Reward / Active Agent"),
                ("shaping_mean", "Shaping / Environment"),
            ):
                axes[1, x_index].plot(
                    [row.get(x_name, np.nan) for row in updates],
                    [row.get(name, np.nan) for row in updates],
                    label=f"{run_label}: {label}",
                )
            panels = sorted(
                {
                    (
                        str(row.get("panel_digest", "Unknown Panel")),
                        _red_zone_label(row.get("red_zone_depth")),
                    )
                    for row in results
                }
            )
            for panel, validation_depth in panels:
                selected = sorted(
                    (
                        row
                        for row in results
                        if str(row.get("panel_digest", "Unknown Panel")) == panel
                        and _red_zone_label(row.get("red_zone_depth"))
                        == validation_depth
                        and row.get("purpose") in ("routine", "initialization")
                    ),
                    key=lambda row: row.get("env_steps", 0),
                )
                if selected:
                    x = [
                        row.get(x_name) if row.get(x_name) is not None else np.nan
                        for row in selected
                    ]
                    scores = [
                        row.get("score") if row.get("complete") else np.nan
                        for row in selected
                    ]
                    plotted = axes[0, x_index].plot(
                        x,
                        scores,
                        marker="o",
                        label=(
                            f"{run_label}\nPanel {panel[:8]} / "
                            f"Validation Red Zone: {validation_depth}"
                        ),
                    )
                    lower = [
                        row.get("ci_low")
                        if row.get("complete") and row.get("ci_low") is not None
                        else np.nan
                        for row in selected
                    ]
                    upper = [
                        row.get("ci_high")
                        if row.get("complete") and row.get("ci_high") is not None
                        else np.nan
                        for row in selected
                    ]
                    axes[0, x_index].vlines(
                        x, lower, upper, colors=plotted[0].get_color(), alpha=0.4
                    )
                    axes[0, x_index].fill_between(
                        x,
                        lower,
                        upper,
                        alpha=0.15,
                    )
        summaries.append(summary)
    for x_index, label in enumerate(
        ("Environment Transitions", "Recorded Active Time (Seconds)")
    ):
        for row_index in range(2):
            axis = axes[row_index, x_index]
            axis.set_xlabel(label)
            axis.set_ylabel(
                "Fixed-Panel Expected Score"
                if row_index == 0
                else "Mean Reward (Separate Denominators)"
            )
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize="small")
        axes[0, x_index].set_ylim(-0.02, 1.02)
        if not any(
            np.isfinite(np.asarray(line.get_ydata(), dtype=float)).any()
            for line in axes[0, x_index].lines
        ):
            axes[0, x_index].text(
                0.5,
                0.5,
                "No Validation Results",
                ha="center",
                va="center",
                transform=axes[0, x_index].transAxes,
                color="0.4",
            )
    paths: dict[str, str] = {}
    for suffix in ("png", "svg"):
        target = destination / f"learning_curves.{suffix}"
        temporary = destination / f".learning_curves.{suffix}.tmp"
        figure.savefig(temporary, format=suffix, dpi=150)
        os.replace(temporary, target)
        paths[suffix] = str(target)
    plt.close(figure)
    if qmix_runs:
        paths.update(_qmix_figure(qmix_runs, destination, plt))
    if pqn_runs:
        paths.update(_pqn_figure(pqn_runs, destination, plt))
    for key, name, rows in (
        ("curve", "learning_curve.csv", curves),
        ("cells", "validation_cells.csv", cells),
        ("exposure", "exposure_history.csv", exposure_history),
    ):
        target = destination / name
        _atomic_text(target, _csv(rows))
        paths[key] = str(target)
    paths["summary"] = str(_write_summary(summaries, destination))
    variation = None
    curve_note: str | None = None
    if declared_grouping is not None:
        variation = _training_variation(
            declared_grouping, summaries, details_by_path, point_cells
        )
        paths.update(_write_training_variation(variation, destination))
        curve_paths, curve_note = _write_recipe_curves(
            declared_grouping, curve_inputs, destination, plt
        )
        paths.update(curve_paths)
        summary_path = destination / "run_summary.md"
        _atomic_text(
            summary_path,
            summary_path.read_text() + f"\n## Recipe Learning Curves\n\n{curve_note}\n",
        )
    return {
        "artifacts": paths,
        "runs": summaries,
        **(
            {"selection_decision": selection_reference}
            if selection_reference is not None
            else {}
        ),
        **({"training_variation": variation} if variation is not None else {}),
        **({"recipe_curves": curve_note} if curve_note is not None else {}),
        "complete": all(row["status"] == "complete" for row in summaries)
        and (
            variation is None
            or all(row["status"] == "complete" for row in variation["runs"])
        ),
        "qualification": (
            "Provisional; inspect saved evidence before making learning claims"
        ),
    }


# Stable columns keep absent evidence distinct from an omitted table. Detailed
# saved values stay in records_json rather than inventing per-method meanings.
_STUDY_TABLES: dict[str, tuple[str, ...]] = {
    "run_index.csv": (
        "study_id",
        "declaration_sha256",
        "case_id",
        "seed",
        "branch_seed",
        "attempt_id",
        "attempt_number",
        "run_attempt_id",
        "run_id",
        "stage",
        "method",
        "recipe",
        "lineage_root_run_id",
        "parent_run_id",
        "status",
        "reason",
        "start_env_steps",
        "requested_additional_env_steps",
        "planned_env_steps",
        "actual_env_steps",
        "device",
        "source_id",
        "dependency_id",
        "runtime_identity_scope",
        "study_path",
        "run_path",
        "latest_checkpoint_path",
        "selected_actor_path",
        "final_actor_path",
    ),
    "learning/training_updates.csv": (
        "study_id",
        "case_id",
        "run_id",
        "attempt_id",
        "run_attempt_id",
        "update",
        "env_steps",
        "acceptance",
        "training_seconds",
        "elapsed_seconds",
        "wall_seconds",
        "records_json",
    ),
    "learning/validation_cells.csv": (
        "study_id",
        "case_id",
        "run_id",
        "checkpoint_id",
        "task_id",
        "recorded_run_attempt_id",
        "execution_run_attempt_ids_json",
        "purpose",
        "env_steps",
        "map_id",
        "opponent",
        "score",
        "mean_team_a_score",
        "mean_team_b_score",
        "records_json",
    ),
    "learning/exposure_history.csv": (
        "study_id",
        "case_id",
        "run_id",
        "attempt_id",
        "run_attempt_id",
        "env_steps",
        "acceptance",
        "records_json",
    ),
    "hyperparameters/design.csv": (
        "study_id",
        "case_id",
        "stage",
        "recipe",
        "factor",
        "level",
        "coding",
        "scale",
        "role",
    ),
    "costs/attempt_costs.csv": (
        "study_id",
        "case_id",
        "attempt_id",
        "attempt_number",
        "run_attempt_id",
        "status",
        "reason",
        "training_seconds",
        "elapsed_seconds",
        "wall_seconds",
        "attempt_training_seconds",
        "attempt_active_seconds",
        "setup_seconds",
        "compile_seconds",
        "output_bytes",
        "validation_seconds",
        "checkpoint_seconds",
        "report_seconds",
        "device",
        "memory_json",
        "measurement_scope",
        "resources_json",
    ),
    "selection/checkpoints.csv": (
        "study_id",
        "case_id",
        "run_id",
        "role",
        "decision_id",
        "checkpoint_id",
        "actor_digest",
        "actor_path",
        "rule",
        "purpose",
        "env_steps",
        "score",
        "complete",
        "records_json",
    ),
    "selection/recipes.csv": (
        "decision_id",
        "name",
        "group",
        "winner",
        "status",
        "measure",
        "units",
        "mean",
        "available_mean",
        "declared_units",
        "available_units",
        "eligible",
        "reasons_json",
        "sources_json",
        "decision_json",
    ),
    "evaluation/games.csv": (
        "evaluation_id",
        "run_id",
        "phase",
        "pass_id",
        "episode_id",
        "system_id",
        "opponent_id",
        "map_id",
        "config_id",
        "seed_id",
        "spawn_locations",
        "paired_comparison_key",
        "bootstrap_group",
        "outcome",
        "team_a_score",
        "team_b_score",
        "episode_length",
    ),
    "evaluation/conditions.csv": (
        "evaluation_id",
        "run_id",
        "phase",
        "pass_id",
        "system_id",
        "opponent_id",
        "status",
        "conditions_json",
        "configurations_json",
        "systems_json",
    ),
    "behavior/metrics.csv": (
        "evaluation_id",
        "run_id",
        "phase",
        "pass_id",
        "episode_id",
        "table",
        "category",
        "metric",
        "value",
        "units",
        "source",
        "recipient",
    ),
    "behavior-evidence/replays.csv": (
        "evaluation_id",
        "run_id",
        "phase",
        "pass_id",
        "episode_id",
        "path",
        "reason",
    ),
}


def _export_csv(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> int:
    """Write one temporary table with fixed columns and return its row count.

    The caller supplies a private staging path and later publishes the completed
    set. Iterate once with bounded row memory; structured values use strict JSON.
    Missing values stay empty, distinct from measured zero. No raw input changes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    json.dumps(row.get(key), sort_keys=True, allow_nan=False)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key)
                    for key in columns
                ]
            )
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return count


def _export_update_rows(
    root: Path, *, include_archives: bool = True
) -> Iterable[Record]:
    """Read accepted updates and labelled suffixes without recovering a writer.

    The saved latest learner description supplies the accepted byte prefix.
    Validate it with the shared log owner before reading. Abandoned attempt
    suffixes remain observed work, never accepted transitions. Historical runs
    without a pointer retain rows labelled unknown. Torn suffixes get an explicit
    record; a malformed accepted row raises. Open recovery withholds acceptance.
    include_archives defaults to True; False reads only the active log for plots.
    """
    from marl_battlegrounds.training._checkpoint_metadata import read_resume_checkpoint
    from marl_battlegrounds.training._run_io import validate_log_cursor

    path = root / "training_updates.jsonl"
    accepted: int | None = None
    if (root / "latest_checkpoint.json").exists():
        _, saved = read_resume_checkpoint(root)
        details = _read_json(root / "run_details.json", {})
        if saved["metadata"].get("run_id") != details.get("run_id"):
            raise ValueError("Accepted update checkpoint belongs to another run")
        cursor = saved["metadata"].get("log_cursors", {}).get(path.name)
        if cursor is not None:
            validate_log_cursor(path, cursor)
            accepted = cursor["size_bytes"]
    recovering = (root / "checkpoint_recovery.json").exists()
    files = ([path] if path.exists() else []) + (
        sorted(
            path
            for path in (root / "attempts").glob("*/*.jsonl")
            if len(path.stem) == 64
            and all(char in "0123456789abcdef" for char in path.stem)
        )
        if include_archives
        else []
    )
    for source in files:
        if source != path:
            with source.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != source.stem:
                    raise ValueError(
                        "Superseded update file differs from its content hash"
                    )
        offset = 0
        with source.open("rb") as stream:
            for line in stream:
                offset += len(line)
                status = (
                    "Superseded"
                    if source != path
                    else "Recovery pending"
                    if recovering
                    else "Unknown historical boundary"
                    if accepted is None
                    else "Accepted"
                    if offset <= accepted
                    else "Uncommitted"
                )
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not line.endswith(b"\n"):
                        raise ValueError("Incomplete training update")
                    json.dumps(row, allow_nan=False)
                except (ValueError, UnicodeError):
                    if status == "Accepted":
                        raise ValueError("Invalid accepted training update") from None
                    row = {
                        "unavailable_reason": "Incomplete observed update",
                        "source_path": str(source),
                    }
                yield {**row, "acceptance": status}


def _unstarted_run(path: Path) -> bool:
    """Recognize an absent or empty pre-admission run directory.

    Only explicit grouping callers use this unavailable-case rule. Empty
    includes a folder holding only the trainer's ``.training.lock``, the shared
    _run_io.unused_run_folder rule. Any other directory with missing run
    details remains damaged evidence, not an omitted run. File, permission
    and malformed-metadata errors are never suppressed.
    """
    from marl_battlegrounds.training._run_io import unused_run_folder

    return unused_run_folder(path)


def _checked_recipe_report(
    report: Mapping[str, Any],
    allowed_runs: set[str],
    *,
    source_files: list[Record] | None = None,
    checked_selection: tuple[Record, dict[str, Record]] | None = None,
    point_cells: dict[str, Record] | None = None,
) -> Record | None:
    """Rebuild a supplied grouped report from saved sources before exporting it.

    report is the ordinary analyze return value, not an unverified winner table.
    Reuse its declared grouping and optional verified re-selection owner. Reject
    changed winners, cells, source runs or derived decision files before output.
    Return its checked recipe record, or None for reports without grouping.
    source_files optionally receives checked source identities. checked_selection
    and point_cells reuse one completed selection read within this export call;
    omitted values use the same owners directly, never trust unchecked winners.
    """
    variation = report.get("training_variation")
    if variation is None:
        if report.get("recipe_selection") is not None:
            raise ValueError("Recipe outputs need their full analyze report")
        return None
    if not isinstance(variation, dict) or not isinstance(
        variation.get("declaration"), dict
    ):
        raise ValueError("Grouped reports need their saved declaration")
    variation = cast(Record, variation)
    declaration = cast(Record, variation["declaration"])
    paths = [Path(row["run_dir"]).resolve() for row in declaration["runs"]]
    if not {str(path) for path in paths} <= allowed_runs:
        raise ValueError("Analysis report names a run outside the declared studies")
    grouping = _training_grouping(declaration, paths)
    summaries = [_summary(path) for path in paths if not _unstarted_run(path)]
    details = {
        str(path): _read_json(path / "run_details.json", {})
        for path in paths
        if not _unstarted_run(path)
    }
    if point_cells is None and grouping.get("measure") == "point_margin":
        point_cells = {}
    if report.get("selection_decision") is not None:
        reference, selected = checked_selection or _analysis_selection(
            report["selection_decision"]["path"],
            paths,
            point_cells=point_cells,
            source_files=source_files,
        )
        for summary in summaries:
            summary.update(
                selection=selected[summary["run_dir"]]["winner"],
                selection_decision=reference,
            )
    elif point_cells is not None:
        point_cells.update(_recover_selection_points(paths, source_files=source_files))
    rebuilt = _training_variation(grouping, summaries, details, point_cells)
    if rebuilt != variation:
        raise ValueError("Analysis report differs from its saved run evidence")
    result = rebuilt.get("recipe_selection")
    artifact = report.get("artifacts", {}).get("recipe_selection")
    if artifact is not None and _read_json(Path(artifact), None) != result:
        raise ValueError("Recipe decision file differs from its checked report")
    return result


def _read_data_index(output_dir: Path) -> Record:
    """Read a completed derived table set, refusing interrupted or changed files.

    Verify only the exact table paths and hashes named by data_index.json. Raw
    sources need not remain online to inspect this exported snapshot. Relative
    table paths must stay inside output_dir and cannot follow links. An interrupted
    refresh can leave mixed versions; the caller must refresh from saved sources.
    """
    index = _read_json(output_dir / "data_index.json", None)
    if not isinstance(index, dict) or index.get("schema_version") != 1:
        raise ValueError("Study data index is missing or unsupported; export it again")
    index = cast(Record, index)
    expected = {*_STUDY_TABLES, "hyperparameters/configurations.jsonl"}
    if not isinstance(index.get("tables"), dict) or set(index["tables"]) != expected:
        raise ValueError("Study data index has missing or unknown tables")
    for name, description in index["tables"].items():
        path = output_dir / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.resolve() != path.absolute()
            or not path.is_file()
        ):
            raise ValueError("Study data index has a missing or unsafe table")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != description.get("sha256"):
            raise ValueError("Study data tables are inconsistent; export them again")
    return index


def load_study_data(output_dir: str | Path) -> dict[str, Path]:
    """Return checked paths to a previously exported study table set.

    Parameters
    ----------
    output_dir : path
        Folder written by export_study_data. Moving the whole folder is supported.
        Source runs need not be online to read this derived snapshot.

    Returns
    -------
    dict[str, pathlib.Path]
        Absolute paths keyed by relative table names, plus data_index.json. Use
        ordinary CSV/JSON readers after this check. The index describes units,
        coverage and named source roots; no models or numerical backend is loaded.

    Raises
    ------
    ValueError, OSError
        A file is missing, linked outside the folder, changed or from an interrupted
        export. Refresh with export_study_data before using an inconsistent set.

    Notes
    -----
    Checks every exported table hash once, with bounded read memory. It does not
    revalidate raw evidence, run analysis or write files. A check cannot protect
    against another process changing tables afterward; use a settled output folder.
    """
    root = Path(output_dir).expanduser().resolve()
    index = _read_data_index(root)
    return {name: root / name for name in (*index["tables"], "data_index.json")}


def _export_paths(value: object, roots: dict[str, Path], field: str = "") -> object:
    """Replace absolute saved path values by named-root references recursively.

    roots maps stable export-local names to original directories; add an external
    path's parent only when no declared root contains it. Returned JSON values
    keep historical records portable without rewriting their raw source bytes.
    field names the containing JSON key (empty at the root). It identifies
    external path fields; unrelated text stays unchanged. Named roots are later
    written relative to the chosen analysis folder. Plain
    strings without an absolute path remain unchanged. No files are opened.
    """
    if isinstance(value, dict):
        return {
            key: _export_paths(item, roots, key)
            for key, item in cast(Record, value).items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _export_paths(item, roots, field) for item in cast(Sequence[Any], value)
        ]
    if isinstance(value, (str, Path)) and str(value).startswith("/"):
        path = Path(value)
        for name, root in sorted(roots.items(), key=lambda item: -len(item[1].parts)):
            if path.is_relative_to(root):
                return f"{name}/{path.relative_to(root).as_posix()}"
        if not isinstance(value, Path) and not (
            field.endswith(
                ("_path", "_paths", "_dir", "_dirs", "_checkpoint", "_actor")
            )
            or field
            in {
                "path",
                "checkpoint",
                "directory",
                "reference",
                "pinned_opponent",
                "initial_actor",
                "validation_panel",
                "opponent_pool",
            }
        ):
            return value
        name = f"source_{len(roots) + 1}"
        roots[name] = path.parent
        return f"{name}/{path.name}"
    return value


def _attempt_export_costs(context: Record, attempt: Record) -> Record:
    """Join observed costs only by an exact admitted study/runner attempt binding.

    context contains one saved case and its event rows. A UUID/request pair must
    identify exactly one runner attempt. Missing legacy bindings remain unknown;
    they never borrow a previous status. Clocks remain cumulative facts, with
    separate within-attempt differences only when a saved baseline exists.
    """
    identifier = attempt.get("study_attempt_id")
    digest = attempt.get("request_sha256")
    expected = {
        "study_attempt_id": identifier,
        "request_sha256": digest,
        "case_id": context.get("job", {}).get("id"),
        "number": attempt.get("number"),
        "declaration_sha256": context.get("declaration_sha256"),
    }

    def matches(row: Record) -> bool:
        """Check a saved row against this case and exact worker request.

        Older records with no UUID/request binding remain unavailable. Extra
        case fields are checked whenever the calling context declares them.
        This comparison reads no files and does not change the row.
        """
        binding: Record = row.get("study_attempt") or {}
        return (
            identifier is not None
            and digest is not None
            and all(
                value is None or binding.get(name) == value
                for name, value in expected.items()
            )
        )

    events = [row for row in context["events"] if matches(row)]
    saved_status: Record = attempt.get("run_status") or {}
    if not matches(saved_status):
        saved_status = context.get("status") or {}
    if not matches(saved_status):
        saved_status = {}
    owners = {
        row.get("attempt_id")
        for row in [*events, saved_status]
        if row.get("attempt_id") is not None
    }
    if len(owners) > 1:
        raise ValueError("One study attempt names conflicting runner attempts")
    runner_id = next(iter(owners), attempt.get("run_attempt_id"))
    if owners and attempt.get("run_attempt_id", runner_id) != runner_id:
        raise ValueError("Study and runner attempt identities disagree")
    starts = [row for row in events if row.get("event") == "attempt_started"]
    if len(starts) > 1:
        raise ValueError("Runner attempt has duplicate start records")
    start: Record = starts[0] if starts else {}
    terminal = next(
        (
            row
            for row in reversed(events)
            if row.get("event") in ("attempt_complete", "attempt_failed")
        ),
        cast(Record, {}),
    )
    latest: Record = terminal or saved_status or (events[-1] if events else {})
    phases = _phase_costs(events, {})
    result: Record = {
        "run_attempt_id": runner_id,
        "actual_env_steps": latest.get("env_steps"),
        "training_seconds": latest.get("training_seconds"),
        "elapsed_seconds": latest.get("elapsed_seconds"),
        "wall_seconds": latest.get("wall_seconds"),
        "attempt_training_seconds": None,
        "attempt_active_seconds": None,
        "setup_seconds": phases["Setup"]["seconds"],
        "validation_seconds": phases["Validation"]["seconds"],
        "checkpoint_seconds": phases["Checkpoint"]["seconds"],
        "report_seconds": phases["Report"]["seconds"],
        "memory_json": latest.get("memory"),
        "measurement_scope": "Observed saved attempt facts"
        if events or saved_status
        else "Unavailable: No exact runner-attempt binding",
        "resources_json": attempt.get("resource_facts", attempt.get("resources")),
        "device": attempt.get("resources", {}).get("device"),
    }
    for clock, baseline, output in (
        ("training_seconds", "training_start_seconds", "attempt_training_seconds"),
        ("elapsed_seconds", "elapsed_start_seconds", "attempt_active_seconds"),
    ):
        first, last = start.get(baseline), latest.get(clock)
        if (
            isinstance(first, (int, float))
            and not isinstance(first, bool)
            and isinstance(last, (int, float))
            and not isinstance(last, bool)
            and math.isfinite(first)
            and math.isfinite(last)
            and last >= first
        ):
            result[output] = last - first
    return result


def export_study_data(
    study_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    reports: Sequence[Mapping[str, Any]] = (),
    evaluation_dirs: Sequence[str | Path] = (),
    protocol: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Export saved study, attempt and evaluation tables without running methods.

    Parameters
    ----------
    study_dirs : sequence of paths
        Exact study directories, in declared order. Include every intended study;
        all planned cases and attempts appear, even when no run folder exists.
    output_dir : path
        Chosen derived-data folder. It cannot replace a study or raw run/evaluation
        folder. Existing derived tables are refreshed; raw inputs never change.
    reports : sequence of mappings, default=()
        Complete return values from training.analyze for these studies. Grouped
        recipe decisions are rebuilt against saved evidence before export. A
        hand-written winner or changed decision file is rejected.
    evaluation_dirs : sequence of paths, default=()
        Exact saved evaluation roots. Read public result tables in bounded batches;
        preserve original game, map, spawn, policy, metric and replay ownership.
        No evaluation, missing metric or replay is reconstructed.
    protocol : JSON/YAML path, mapping or None, default=None
        Optional schema_version=1 record with cases and factors lists. YAML
        needs the hydra extra. File-relative study paths resolve beside that
        declaration; mapping paths use cwd. Each row
        names an exact study_dir and expanded case_id (including its seed).
        Cases additionally name stage and recipe. Factors additionally name
        factor, level, coding, scale and role (fixed or searched). These are
        explicit paper declarations, never inferred from settings or outcomes.

    Returns
    -------
    dict[str, pathlib.Path]
        Absolute paths keyed by the relative file names, including data_index.json.
        Every required CSV has fixed headers even when empty. The index records
        row counts, hashes, source roots and unavailable measurements. It is
        published last. Path references use named source roots, so moving a
        complete bundle together retains their meaning.

    Raises
    ------
    ValueError, OSError
        Inputs, identities, accepted log prefixes or report evidence are invalid,
        source/output paths collide, or files change during export. All inputs
        are read into staged tables before any old derived file is replaced.
        A process interruption during publication may leave mixed derived files;
        their old index will fail verification until this function refreshes them.

    Notes
    -----
    Host-only. No learner, device query, provider, environment or recovery writer
    runs. Historical missing attempt bindings/costs stay unknown. Cumulative
    training, active and wall clocks are not summed across attempts. Full metric
    rows stream in batches; they are never collected into one wide DataFrame.
    A successful export does not establish learning or scientific reproducibility.
    """
    from marl_battlegrounds.training import study
    from marl_battlegrounds.training._checkpoint_metadata import (
        _artifact_identity,  # pyright: ignore[reportPrivateUsage]
        read_checkpoint_description,
        read_resume_checkpoint,
    )
    from marl_battlegrounds.training._run_io import atomic_json, saved_result_actors

    if not study_dirs:
        raise ValueError("export_study_data needs at least one exact study")
    roots = [Path(value).expanduser().resolve() for value in study_dirs]
    evaluations = [Path(value).expanduser().resolve() for value in evaluation_dirs]
    destination = Path(output_dir).expanduser().absolute()
    if destination.resolve() != destination:
        raise ValueError("Study export destination must not follow links")
    if len(set(roots)) != len(roots) or len(set(evaluations)) != len(evaluations):
        raise ValueError("Study and evaluation roots must be distinct within each list")
    if any(
        root == destination or root.is_relative_to(destination)
        for root in (*roots, *evaluations)
    ):
        raise ValueError("Derived output cannot replace or contain a raw input root")
    locations = {f"study_{i + 1}": root for i, root in enumerate(roots)}
    locations.update(
        {f"evaluation_{i + 1}": root for i, root in enumerate(evaluations)}
    )
    sources: dict[Path, str] = {}

    def remember(path: Path, expected: str | None = None) -> None:
        """Bind input bytes or a checked reader digest; reject change/disappearance."""
        if not path.exists():
            if path in sources or expected is not None:
                raise ValueError(f"Export source disappeared while reading: {path}")
            return
        if path.resolve() != path or not path.is_file():
            raise ValueError(f"Export source must be a regular file: {path}")
        if expected is None:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            digest = expected
        if path in sources and sources[path] != digest:
            raise ValueError(f"Export source changed while reading: {path}")
        sources[path] = digest

    contexts: list[Record] = []
    for index, root in enumerate(roots):
        for name in ("declaration.json", "study.json", "report.json"):
            remember(root / name)
        declared = study._read(root / "declaration.json")  # pyright: ignore[reportPrivateUsage]
        status = study.study_status(root)
        study_id = f"study_{index + 1}"
        for job in study._jobs(declared):  # pyright: ignore[reportPrivateUsage]
            run = root / "cases" / job["id"] / "run"
            if destination == run or destination.is_relative_to(run):
                raise ValueError("Derived tables cannot be written inside a raw run")
            case = status["study"].get("cases", {}).get(job["id"], {})
            if run.exists():
                for path in (
                    *run.glob("*.json"),
                    *run.glob("*.jsonl"),
                    *run.glob("attempts/*/*.jsonl"),
                    *run.glob("checkpoints/*/checkpoint_details.json"),
                    *run.glob("actors/*/actor_details.json"),
                ):
                    remember(path)
            details = _read_json(run / "run_details.json", {})
            if not isinstance(details, dict):
                raise ValueError("Saved run details must be an object")
            details = cast(Record, details)
            base_config = job.get("config", {})
            start_steps = 0
            parent_metadata: Record = {}
            if job["kind"] == "extend":
                from marl_battlegrounds.training.config import config_to_dict
                from marl_battlegrounds.training.runner import (
                    _saved_config,  # pyright: ignore[reportPrivateUsage]
                )

                parent_path = Path(job["checkpoint"]).resolve()
                remember(parent_path / "checkpoint_details.json")
                parent = read_checkpoint_description(parent_path)
                expected = job["parent_identity"]
                if (
                    parent["kind"] != "learner"
                    or parent["checkpoint_id"] != expected["checkpoint_id"]
                    or study._digest(parent) != expected["description_sha256"]  # pyright: ignore[reportPrivateUsage]
                ):  # pyright: ignore[reportPrivateUsage]
                    raise ValueError(
                        "Study continuation parent differs from its declared identity"
                    )
                parent_continuation = parent["metadata"].get("continuation")
                if parent_continuation is not None:
                    remember(
                        Path(parent_continuation["parent_checkpoint"])
                        / "checkpoint_details.json"
                    )
                base_config = config_to_dict(_saved_config(parent))
                parent_metadata = parent["metadata"]
                start_steps = parent["counters"]["env_steps"]
            additional_steps = job.get(
                "additional_env_steps", base_config.get("total_env_steps")
            )
            run_status = _read_json(run / "status.json", {})
            final_actor = selected_actor = None
            final_description = None
            if (
                run_status.get("status") == "complete"
                and not (run / "checkpoint_recovery.json").exists()
                and (
                    (run / "latest_checkpoint.json").exists()
                    or run_status.get("final_actor") is not None
                    or run_status.get("selected_actor") is not None
                )
            ):
                checkpoint, latest = read_resume_checkpoint(run)
                remember(checkpoint / "checkpoint_details.json")
                _, final_actor, selected_actor = saved_result_actors(
                    run,
                    latest,
                    run_status,
                    run_details=details,
                    record_source=remember,
                )
                if final_actor is not None:
                    final_description = read_checkpoint_description(final_actor)
            method = details.get("config", base_config).get(
                "method", job.get("parent_identity", {}).get("method", "mappo")
            )
            contexts.append(
                {
                    "study_id": study_id,
                    "study_dir": root,
                    "declaration_sha256": status["declaration_sha256"],
                    "job": job,
                    "case": case,
                    "run": run,
                    "details": details,
                    "base_config": base_config,
                    "parent_metadata": parent_metadata,
                    "start_env_steps": start_steps,
                    "additional_env_steps": additional_steps,
                    "planned_env_steps": start_steps + additional_steps
                    if additional_steps is not None
                    else None,
                    "status": run_status,
                    "final_actor": final_actor,
                    "selected_actor": selected_actor,
                    "final_description": final_description,
                    "declared_runtime": declared.get("runtime_identities", {}).get(
                        method, {}
                    ),
                    "events": _read_events(run / "run_events.jsonl"),
                }
            )
    for context in contexts:
        context["attempt_facts"] = [
            (attempt, _attempt_export_costs(context, attempt))
            for attempt in cast(
                list[Record], context["case"].get("attempts", []) or [{}]
            )
        ]
    allowed_runs = {str(row["run"]) for row in contexts}
    grouped_recipes: dict[str, str] = {}
    recipes: list[Record] = []
    reselections: dict[str, list[Record]] = {}
    for index, report in enumerate(reports):
        for raw in report.get("artifacts", {}).values():
            path = Path(raw).resolve()
            locations.setdefault(f"report_{index + 1}", path.parent)
            remember(path)
        checked_sources: list[Record] = []
        reference = report.get("selection_decision")
        checked_selection = None
        point_cells: dict[str, Record] | None = (
            {}
            if report.get("training_variation", {})
            .get("declaration", {})
            .get("measure")
            == "point_margin"
            else None
        )
        if reference is not None:
            selected_paths = [row["run_dir"] for row in report.get("runs", [])]
            if not set(selected_paths) <= allowed_runs:
                raise ValueError(
                    "Selection report names a run outside the declared studies"
                )
            checked_selection = _analysis_selection(
                reference["path"],
                selected_paths,
                point_cells=point_cells,
                source_files=checked_sources,
            )
            checked, choices = checked_selection
            if checked != reference:
                raise ValueError(
                    "Selection report identity differs from its saved decision"
                )
            remember(Path(checked["path"]))
            for path, choice in choices.items():
                reselections.setdefault(path, []).append(
                    {**choice, "decision": checked}
                )
        recipe = _checked_recipe_report(
            report,
            allowed_runs,
            checked_selection=checked_selection,
            point_cells=point_cells,
            source_files=checked_sources,
        )
        if report.get("training_variation") is not None:
            for row in report["training_variation"]["declaration"]["runs"]:
                path = str(Path(row["run_dir"]).resolve())
                if path in grouped_recipes and grouped_recipes[path] != row["group"]:
                    raise ValueError("Reports assign conflicting recipes to one run")
                grouped_recipes[path] = row["group"]
        if recipe is not None:
            recipes.append(recipe)
        for source in checked_sources:
            remember(Path(source["path"]), source["sha256"])
    protocol_base = Path.cwd()
    if isinstance(protocol, (str, Path)):
        protocol_path = Path(protocol).expanduser().resolve()
        protocol_base = protocol_path.parent
        declarations = read_declaration(protocol_path)
    else:
        declarations = {} if protocol is None else dict(protocol)
    if declarations and (
        declarations.get("schema_version") != 1
        or set(declarations) - {"schema_version", "cases", "factors"}
    ):
        raise ValueError("Protocol needs schema_version 1, cases and factors only")
    roles: dict[tuple[str, str], Record] = {}
    factors: list[Record] = []
    known_cases = {
        (str(row["study_dir"]), row["job"]["id"]): row["study_id"] for row in contexts
    }
    for name in ("cases", "factors"):
        values = declarations.get(name, [])
        if not isinstance(values, list):
            raise ValueError(f"Protocol {name} must be a list")
        for value in cast(list[Any], values):
            required = {"study_dir", "case_id"} | (
                {"stage", "recipe"}
                if name == "cases"
                else {"factor", "level", "coding", "scale", "role"}
            )
            if not isinstance(value, dict) or set(cast(Record, value)) != required:
                raise ValueError(f"Protocol {name} row has missing or unknown fields")
            value = cast(Record, value)
            value = {
                **value,
                "study_dir": str(
                    (protocol_base / Path(value["study_dir"]).expanduser()).resolve()
                ),
            }
            key = (value["study_dir"], value["case_id"])
            if key not in known_cases:
                raise ValueError("Protocol names a case outside the declared studies")
            if name == "cases":
                if key in roles:
                    raise ValueError("Protocol repeats a case role")
                run_path = str(Path(key[0]) / "cases" / key[1] / "run")
                if (
                    run_path in grouped_recipes
                    and grouped_recipes[run_path] != value["recipe"]
                ):
                    raise ValueError(
                        "Protocol recipe differs from the checked grouping"
                    )
                roles[key] = value
            else:
                if value["role"] not in ("fixed", "searched"):
                    raise ValueError("Factor role must be fixed or searched")
                factors.append(
                    {
                        **value,
                        "study_id": known_cases[key],
                        "stage": roles.get(key, {}).get("stage"),
                        "recipe": roles.get(key, {}).get(
                            "recipe",
                            grouped_recipes.get(
                                str(Path(key[0]) / "cases" / key[1] / "run")
                            ),
                        ),
                    }
                )
    json.dumps(declarations, allow_nan=False)
    from marl_battlegrounds.evaluation.results import load_results

    result_views: list[tuple[str, Path, Any]] = []
    unavailable: list[str] = []
    for index, root in enumerate(evaluations):
        remember(root / "run_details.json")
        result = load_results(root)
        for filename in result.metadata.get("tables", {}).values():
            for name in filename.get("files", ()):
                remember(root / name)
        # The manifest is the file authority; table metadata describes projections.
        manifest = _read_json(root / "run_details.json", {})
        for name in manifest.get("tables", {}):
            remember(root / name)
        for replay in result.replay_paths:
            remember(replay)
        result_views.append((f"evaluation_{index + 1}", root, result))
    if not evaluations:
        unavailable.append(
            "No evaluation roots were supplied; games, metrics "
            "and replays are unavailable"
        )
    if not factors:
        unavailable.append("No explicit factor design was supplied")
    if not reports:
        unavailable.append("No separate analysis or selection reports were supplied")

    def rows_for(name: str) -> Iterable[Record]:
        """Yield one requested table directly from the already admitted inputs."""
        if name == "hyperparameters/design.csv":
            yield from factors
            return
        if name == "selection/recipes.csv":
            for recipe in recipes:
                for decision in recipe["decisions"]:
                    for candidate in decision["candidates"]:
                        yield {
                            **candidate,
                            "decision_id": recipe["content_id"],
                            "name": decision["name"],
                            "winner": decision["winner"],
                            "status": decision["status"],
                            "measure": recipe["measure"],
                            "units": recipe["units"],
                            "reasons_json": candidate["unavailable_reasons"],
                            "sources_json": candidate["sources"],
                            "decision_json": {**recipe, "decisions": [decision]},
                        }
            return
        if name.startswith(("evaluation/", "behavior/", "behavior-evidence/")):
            yield from _evaluation_export_rows(
                name, result_views, destination, unavailable
            )
            return
        for context in contexts:
            job, case, run, details = (
                context[key] for key in ("job", "case", "run", "details")
            )
            role = roles.get((str(context["study_dir"]), job["id"]), {})
            config = details.get("config", context["base_config"])
            common = {
                "study_id": context["study_id"],
                "case_id": job["id"],
                "run_id": details.get("run_id"),
            }
            if name in ("run_index.csv", "costs/attempt_costs.csv"):
                for attempt, costs in context["attempt_facts"]:
                    attempt_status = attempt.get("status")
                    if (
                        attempt
                        and attempt.get("study_attempt_id") is None
                        and attempt is case["attempts"][-1]
                        and case.get("status") in ("complete", "failed", "time_limit")
                        and attempt_status in (None, "running")
                    ):
                        # Old controllers finalized the case without finalizing
                        # its last attempt. This supplies status, never costs.
                        attempt_status = case["status"]
                    yield {
                        **common,
                        **costs,
                        "declaration_sha256": context["declaration_sha256"],
                        "attempt_id": attempt.get("study_attempt_id"),
                        "attempt_number": attempt.get("number"),
                        "seed": config.get("seed"),
                        "branch_seed": details.get("continuation", {})
                        .get("changes", job.get("changes", {}))
                        .get("seed")
                        if job["kind"] == "extend"
                        else None,
                        "method": config.get(
                            "method", job.get("parent_identity", {}).get("method")
                        ),
                        "stage": role.get("stage"),
                        "recipe": role.get("recipe", grouped_recipes.get(str(run))),
                        "lineage_root_run_id": _training_lineage(
                            details or context["parent_metadata"]
                        ),
                        "parent_run_id": details.get("continuation", {}).get(
                            "parent_run_id", context["parent_metadata"].get("run_id")
                        ),
                        "status": attempt_status
                        or (
                            "time_limit"
                            if attempt.get("exit_code") == 124
                            else "failed"
                            if attempt.get("exit_code") not in (None, 0)
                            or attempt.get("error")
                            else "complete"
                            if attempt.get("exit_code") == 0
                            and attempt.get("run_status", {}).get("status")
                            == "complete"
                            else "interrupted"
                            if attempt.get("interrupted")
                            else "running"
                            if attempt
                            else case.get("status", "pending")
                        ),
                        "reason": attempt.get("error")
                        if attempt
                        else case.get("error"),
                        "start_env_steps": context["start_env_steps"],
                        "requested_additional_env_steps": context[
                            "additional_env_steps"
                        ],
                        "planned_env_steps": context["planned_env_steps"],
                        "source_id": details.get(
                            "source", context["declared_runtime"].get("source")
                        ),
                        "dependency_id": details.get(
                            "dependencies",
                            context["declared_runtime"].get("dependencies"),
                        ),
                        "runtime_identity_scope": "Admitted run"
                        if details
                        else "Declared worker",
                        "study_path": context["study_id"],
                        "latest_checkpoint_path": context["status"].get(
                            "latest_checkpoint"
                        ),
                        "selected_actor_path": context["selected_actor"],
                        "final_actor_path": context["final_actor"],
                        "run_path": f"{context['study_id']}/cases/{job['id']}/run",
                    }
            elif name == "learning/training_updates.csv":
                if run.exists():
                    for row in _export_update_rows(run):
                        yield {
                            **common,
                            **row,
                            "attempt_id": next(
                                (
                                    attempt.get("study_attempt_id")
                                    for attempt, facts in context["attempt_facts"]
                                    if facts.get("run_attempt_id") is not None
                                    and facts["run_attempt_id"] == row.get("attempt_id")
                                ),
                                None,
                            ),
                            "run_attempt_id": row.get("attempt_id"),
                            "update": row.get(
                                "update_index",
                                row.get("update", row.get("completed_updates")),
                            ),
                            "records_json": row,
                        }
            elif name == "learning/exposure_history.csv":
                if details:
                    for row in _exposure_history(run):
                        yield {
                            **common,
                            **row,
                            "attempt_id": (row.get("study_attempt") or {}).get(
                                "study_attempt_id"
                            ),
                            "run_attempt_id": row.get("attempt_id"),
                            "env_steps": row.get(
                                "actual_env_steps", row.get("env_steps")
                            ),
                            "acceptance": row.get(
                                "acceptance",
                                "Accepted"
                                if row.get("accepted")
                                else "Uncommitted or superseded",
                            ),
                            "records_json": row,
                        }
            elif name == "learning/validation_cells.csv":
                if not (run / "checkpoint_recovery.json").exists():
                    values = _read_json(run / "validation_results.json", [])
                    if isinstance(values, dict):
                        values = cast(Record, values).get("results", [])
                    for record in cast(list[Record], values):
                        task_events = [
                            event
                            for event in context["events"]
                            if event.get(
                                "task_id", event.get("result", {}).get("task_id")
                            )
                            == record.get("task_id")
                            and record.get("task_id") is not None
                        ]
                        recorded = next(
                            (
                                event.get("attempt_id")
                                for event in reversed(task_events)
                                if event.get("event") == "validation_complete"
                                and event.get("result") == record
                            ),
                            None,
                        )
                        attempts = list(
                            dict.fromkeys(
                                event["attempt_id"]
                                for event in task_events
                                if event.get("attempt_id") is not None
                            )
                        )
                        for cell in record.get("cells", []):
                            yield {
                                **common,
                                **record,
                                **cell,
                                "recorded_run_attempt_id": recorded,
                                "execution_run_attempt_ids_json": attempts,
                                "records_json": {
                                    "validation": {
                                        key: value
                                        for key, value in record.items()
                                        if key != "cells"
                                    },
                                    "cell": cell,
                                },
                            }
            elif (
                name == "selection/checkpoints.csv"
                and not (run / "checkpoint_recovery.json").exists()
            ):
                selected = _read_json(run / "selection.json", None)
                if selected is not None:
                    yield {
                        **common,
                        **selected,
                        "role": "Original selection",
                        "actor_path": context["selected_actor"],
                        "rule": details.get("selection_rule", "saved"),
                        "records_json": selected,
                    }
                for choice in reselections.get(str(run), []):
                    yield {
                        **common,
                        **choice["winner"],
                        "role": "Declared re-selection",
                        "rule": choice["decision"]["rule"],
                        "decision_id": choice["decision"]["decision_id"],
                        "records_json": choice,
                    }
                actor_path = context["final_actor"]
                actor = context["final_description"]
                if actor is not None:
                    yield {
                        **common,
                        "role": "Final actor",
                        "checkpoint_id": actor["metadata"].get("checkpoint_id"),
                        "actor_digest": _artifact_identity(actor)["actor_digest"],
                        "actor_path": actor_path,
                        "env_steps": actor["metadata"].get("env_steps"),
                        "records_json": actor,
                    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    table_index: Record = {}
    with tempfile.TemporaryDirectory(
        prefix=".study-export-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        for name, columns in _STUDY_TABLES.items():
            count = _export_csv(
                staging / name,
                columns,
                (cast(Record, _export_paths(row, locations)) for row in rows_for(name)),
            )
            table_index[name] = {"rows": count, "columns": list(columns)}
        config_name = "hyperparameters/configurations.jsonl"
        with (staging / config_name).open("w", encoding="utf-8") as stream:
            for context in contexts:
                payload = {
                    "study_id": context["study_id"],
                    "case_id": context["job"]["id"],
                    "declaration_sha256": context["declaration_sha256"],
                    "job": context["job"],
                    "base_config": context["details"].get(
                        "config", context["base_config"]
                    ),
                    "configuration_scope": "Saved admitted settings"
                    if context["details"]
                    else "Parent settings and future changes; child not admitted"
                    if context["job"]["kind"] == "extend"
                    else "Declared fresh settings; run not admitted",
                    "continuation": context["details"].get("continuation"),
                    "dependencies": context["details"].get("dependencies"),
                    "runtime": context["details"].get("runtime"),
                    "input_bindings": context["details"].get("input_bindings"),
                }
                payload = cast(Record, _export_paths(payload, locations))
                payload["content_id"] = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, allow_nan=False).encode()
                ).hexdigest()
                stream.write(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        table_index[config_name] = {"rows": len(contexts)}
        for path in tuple(sources):
            remember(path)
        for name, description in table_index.items():
            path = staging / name
            with path.open("rb") as stream:
                description["sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
            description["bytes"] = path.stat().st_size
        source_rows: list[Record] = []
        for path, digest in sorted(sources.items()):
            owner = next(
                (name for name, root in locations.items() if path.is_relative_to(root)),
                None,
            )
            if owner is None:
                _export_paths(path, locations)
                owner = next(
                    name
                    for name, root in locations.items()
                    if path.is_relative_to(root)
                )
            source_rows.append(
                {
                    "root": owner,
                    "path": path.relative_to(locations[owner]).as_posix(),
                    "sha256": digest,
                }
            )
        portable_protocol = _export_paths(declarations, locations)
        data_index: Record = {
            "schema_version": 1,
            "tables": table_index,
            "source_roots": {
                name: os.path.relpath(root, destination)
                for name, root in locations.items()
            },
            "sources": source_rows,
            "unavailable": list(dict.fromkeys(unavailable)),
            "protocol": portable_protocol,
            "qualification": "Saved software records; no new experiments",
        }
        destination.mkdir(parents=True, exist_ok=True)
        for name in table_index:
            target = destination / name
            if (
                target.is_symlink()
                or target.parent.resolve() != target.parent.absolute()
            ):
                raise ValueError("Derived table path must not follow links")
        for name in table_index:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, target)
        atomic_json(destination / "data_index.json", data_index)
    return load_study_data(destination)


def _evaluation_export_rows(
    name: str,
    views: Sequence[tuple[str, Path, Any]],
    destination: Path,
    unavailable: list[str],
) -> Iterable[Record]:
    """Stream explicit saved result tables with their existing metric meanings.

    views are checked public result objects and exact roots. destination anchors
    portable relative replay paths. Append reasons for omitted optional tables;
    never load methods, run games or infer missing measurements.
    """
    from marl_battlegrounds.evaluation.metric_catalog import METRIC_COLUMNS_BY_NAME

    for identifier, root, result in views:
        passes = {
            (entry["phase"], entry["pass_id"]): entry
            for entry in result.metadata["passes"].values()
        }
        common = {"evaluation_id": identifier, "run_id": result.metadata.get("run_id")}
        if name == "evaluation/conditions.csv":
            for (phase, pass_id), entry in passes.items():
                ids = entry.get("system_ids", {})
                yield {
                    **common,
                    "phase": phase,
                    "pass_id": pass_id,
                    "system_id": ids.get("team_a"),
                    "opponent_id": ids.get("team_b"),
                    "status": entry.get("result_state", {}).get("status"),
                    "conditions_json": entry,
                    "systems_json": {
                        key: value
                        for key, value in result.metadata.get("systems", {}).items()
                        if key in ids.values()
                    },
                    "configurations_json": {
                        key: value
                        for key, value in result.metadata.get(
                            "configurations", {}
                        ).items()
                        if any(
                            key == record.get(field)
                            for record in entry.get("episodes", {}).values()
                            for field in (
                                "config_id",
                                "configuration_digest",
                                "source_config_id",
                            )
                        )
                    },
                }
            continue
        if name == "behavior-evidence/replays.csv":
            paths = set(result.replay_paths)
            if not paths:
                unavailable.append(f"{identifier}: No saved Viewer replays")
            for (phase, pass_id), entry in passes.items():
                for episode, record in entry.get("replays", {}).items():
                    path = root / record["path"]
                    if path in paths:
                        yield {
                            **common,
                            "phase": phase,
                            "pass_id": pass_id,
                            "episode_id": episode,
                            "path": os.path.relpath(path, destination),
                            "reason": entry.get("details", {}).get(
                                "replay_selection", "Saved declared replay"
                            ),
                        }
            continue
        full_episodes: set[tuple[str, str, str]] = set()
        tables = (
            ("episodes",)
            if name == "evaluation/games.csv"
            else ("full_metrics", "priority_metrics")
        )
        for table in tables:
            if result.metadata["tables"][table]["availability"] != "available":
                unavailable.append(f"{identifier}: {table} unavailable")
                continue
            for batch in result.iter_table(table, rows=8):
                for index in range(len(next(iter(batch.values())))):
                    row = {
                        key: (
                            value[index].item()
                            if isinstance(value[index], np.generic)
                            else value[index]
                        )
                        for key, value in batch.items()
                    }
                    key = (row["phase"], row["pass_id"], str(row["episode_id"]))
                    if table == "priority_metrics" and key in full_episodes:
                        continue
                    if table == "full_metrics":
                        full_episodes.add(key)
                    entry = passes[row["phase"], row["pass_id"]]
                    episode = entry.get("episodes", {}).get(str(row["episode_id"]), {})
                    if table == "episodes":
                        ids = entry.get("system_ids", {})
                        yield {
                            **common,
                            **row,
                            **{
                                key: episode.get(key)
                                for key in (
                                    "spawn_locations",
                                    "paired_comparison_key",
                                    "bootstrap_group",
                                )
                            },
                            "system_id": ids.get("team_a"),
                            "opponent_id": ids.get("team_b"),
                        }
                    else:
                        identity = {
                            key: row[key]
                            for key in ("run_id", "phase", "pass_id", "episode_id")
                        }
                        for metric, value in row.items():
                            column = METRIC_COLUMNS_BY_NAME.get(metric)
                            if column is None:
                                continue
                            yield {
                                **common,
                                **identity,
                                "table": table,
                                "category": column.family,
                                "metric": metric,
                                "value": value
                                if value is not None and math.isfinite(value)
                                else None,
                                "units": column.unit,
                                "source": {
                                    "scope": column.scope,
                                    "role": column.subject_role,
                                    "subjects": column.subjects,
                                },
                                "recipient": column.recipient_role,
                            }


def _grad_norm_panel(
    axis: Any,  # noqa: ANN401
    runs: Sequence[tuple[str, Sequence[Record]]],
) -> None:
    """Draw saved gradient norms for QMIX or PQN-VDN runs on one figure panel.

    Parameters
    ----------
    axis : matplotlib Axes
        The empty panel to draw on. Its x-axis is environment transitions.
    runs : sequence of (str, sequence of dict)
        Each run's label and its committed update rows. A row may carry
        grad_norm_mean and grad_norm_max (float or None) and
        grad_norm_nonfinite (int, the count of steps in that update whose norm
        was not finite).

    Notes
    -----
    Each run gets a solid mean line and a dashed max line in one color. The
    y-axis is logarithmic when any norm is drawn. None, zero, negative and
    non-finite norms are gaps, never zero. An update with grad_norm_nonfinite
    above 0 gets a red dotted vertical line. Rows without these keys (runs saved
    before gradient norms were logged) add nothing. When no run has a positive
    finite norm or a non-finite count, the panel shows "No Gradient Norms Saved
    In These Runs" instead. The panel sets its own labels, grid and legend.
    Only the given axis changes.
    """
    keys = ("grad_norm_mean", "grad_norm_max", "grad_norm_nonfinite")
    marked = False
    for run_label, rows in runs:
        logged = [row for row in rows if any(key in row for key in keys)]
        if not logged:
            continue
        steps = [row.get("env_steps", np.nan) for row in logged]
        color = None
        for key, style, name in (
            ("grad_norm_mean", "-", "Mean"),
            ("grad_norm_max", "--", "Max"),
        ):
            values = [row.get(key) for row in logged]
            points = [
                float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
                else np.nan
                for value in values
            ]
            if not any(math.isfinite(value) for value in points):
                continue
            (line,) = axis.plot(
                steps, points, style, color=color, label=f"{run_label}: {name}"
            )
            color = line.get_color()
        for row in logged:
            count = row.get("grad_norm_nonfinite")
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                axis.axvline(
                    row.get("env_steps", np.nan),
                    color="tab:red",
                    linestyle=":",
                    alpha=0.6,
                    label=None if marked else "Update With A Non-Finite Norm",
                )
                marked = True
    axis.set_xlabel("Environment Transitions")
    axis.set_ylabel("Gradient Norm (Mean And Max)")
    axis.grid(alpha=0.2)
    if any(line.get_linestyle() != ":" for line in axis.lines):
        axis.set_yscale("log")
    if axis.lines:
        axis.legend(fontsize="small")
    else:
        axis.text(
            0.5,
            0.5,
            "No Gradient Norms Saved In These Runs",
            ha="center",
            va="center",
            transform=axis.transAxes,
            color="0.4",
        )


def _qmix_figure(
    runs: Sequence[tuple[str, Sequence[Record]]],
    destination: Path,
    plt: Any,  # noqa: ANN401
) -> dict[str, str]:
    """Write QMIX learning curves as qmix_curves.png and .svg; return their paths.

    Parameters
    ----------
    runs : sequence of (str, sequence of dict)
        Each QMIX run's label and its committed update rows.
    destination : Path
        The report folder.
    plt : module
        The already-imported matplotlib.pyplot module.

    Returns
    -------
    dict[str, str]
        {"qmix_png": path, "qmix_svg": path}.

    Notes
    -----
    Five panels against environment transitions: loss (warmup blocks are
    gaps), replay rows stored per game, TD pairs used per real transition
    (repeats count again), the exploration rate, and gradient norms (mean and
    max on a log scale, with updates that saw a non-finite norm marked; see
    _grad_norm_panel). Older runs without gradient norms get a short note in
    that panel. Host-only; each file is written to a temporary name and then
    replaced, so only these two figure files change.
    """
    figure, grid = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    axes = grid.flat[:4]
    _grad_norm_panel(grid.flat[4], runs)
    grid.flat[5].set_visible(False)
    panels = (
        ("loss", "Mean Squared Team TD Error"),
        ("replay_rows_per_lane", "Replay Rows Per Game"),
        ("td_pairs_per_transition", "Used TD Pairs Per Transition"),
        ("epsilon", "Exploration Rate"),
    )
    for axis, (name, label) in zip(axes.flat, panels, strict=True):
        for run_label, rows in runs:
            axis.plot(
                [row.get("env_steps", np.nan) for row in rows],
                [
                    np.nan if row.get(name) is None else row.get(name, np.nan)
                    for row in rows
                ],
                label=run_label,
            )
        axis.set_xlabel("Environment Transitions")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize="small")
    paths: dict[str, str] = {}
    for suffix in ("png", "svg"):
        target = destination / f"qmix_curves.{suffix}"
        temporary = destination / f".qmix_curves.{suffix}.tmp"
        figure.savefig(temporary, format=suffix, dpi=150)
        os.replace(temporary, target)
        paths[f"qmix_{suffix}"] = str(target)
    plt.close(figure)
    return paths


def _pqn_figure(
    runs: Sequence[tuple[str, Sequence[Record]]],
    destination: Path,
    plt: Any,  # noqa: ANN401
) -> dict[str, str]:
    """Write PQN-VDN learning curves as pqn_curves.png and .svg; return their paths.

    Parameters
    ----------
    runs : sequence of (str, sequence of dict)
        Each PQN-VDN run's label and its committed update rows.
    destination : Path
        The report folder.
    plt : module
        The already-imported matplotlib.pyplot module.

    Returns
    -------
    dict[str, str]
        {"pqn_png": path, "pqn_svg": path}.

    Notes
    -----
    Five panels against environment transitions: loss (initial chunks are
    gaps), used TD pairs per generated transition (each epoch counts again),
    the share of used pairs that start on kept rows, the exploration rate, and
    gradient norms (mean and max on a log scale, with updates that saw a
    non-finite norm marked; see _grad_norm_panel). Older runs without gradient
    norms get a short note in that panel. All come from scalar row columns.
    Host-only; each file is written to a temporary name and then replaced, so
    only these two figure files change.
    """

    def ratio(row: Record, top: str, bottom: str) -> float:
        """Divide two row counts; missing or zero denominators give NaN."""
        numerator, denominator = row.get(top), row.get(bottom)
        if not isinstance(numerator, (int, float)) or not isinstance(
            denominator, (int, float)
        ):
            return float("nan")
        return float(numerator) / float(denominator) if denominator else float("nan")

    figure, grid = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    axes = grid.flat[:4]
    _grad_norm_panel(grid.flat[4], runs)
    grid.flat[5].set_visible(False)
    panels: tuple[tuple[str, Callable[[Record], float]], ...] = (
        (
            "Mean Squared Team TD Error",
            lambda row: np.nan if row.get("loss") is None else float(row["loss"]),
        ),
        (
            "Used TD Pairs Per Generated Transition",
            lambda row: ratio(row, "used_td_pairs", "env_steps"),
        ),
        (
            "Share Of Used Pairs Starting On Kept Rows",
            lambda row: ratio(row, "used_prefix_td_pairs", "used_td_pairs"),
        ),
        (
            "Exploration Rate",
            lambda row: np.nan if row.get("epsilon") is None else float(row["epsilon"]),
        ),
    )
    for axis, (label, value) in zip(axes.flat, panels, strict=True):
        for run_label, rows in runs:
            axis.plot(
                [row.get("env_steps", np.nan) for row in rows],
                [value(row) for row in rows],
                label=run_label,
            )
        axis.set_xlabel("Environment Transitions")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize="small")
    paths: dict[str, str] = {}
    for suffix in ("png", "svg"):
        target = destination / f"pqn_curves.{suffix}"
        temporary = destination / f".pqn_curves.{suffix}.tmp"
        figure.savefig(temporary, format=suffix, dpi=150)
        os.replace(temporary, target)
        paths[f"pqn_{suffix}"] = str(target)
    plt.close(figure)
    return paths


def _screen_object(path: Path) -> Record:
    """Read an optional screen object; reject non-object JSON without changing it."""
    value = _read_json(path, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return cast(Record, value)


def _screen_seconds(value: object) -> float | None:
    """Return finite nonnegative saved seconds, or None for an unavailable value."""
    if type(value) not in (float, int):
        return None
    number = float(cast(float, value))
    return number if math.isfinite(number) and number >= 0 else None


def _screen_duration(value: object) -> str:
    """Show saved nonnegative seconds as hours, minutes and seconds, or Unknown."""
    from marl_battlegrounds.training._run_io import duration

    seconds = _screen_seconds(value)
    return "Unknown" if seconds is None else duration(seconds)


def _screen_warm_seconds(updates: Sequence[Record], steps: int) -> float | None:
    """Sum recorded blocks through steps, excluding each attempt's first block.

    updates is the committed active training prefix. A missing block time makes
    the sum unknown. The excluded block contains both cold setup and useful
    numerical work; its duration is not a measurement of compilation alone.
    """
    seen: set[object] = set()
    seconds: list[float] = []
    for row in updates:
        if row.get("env_steps", 0) > steps:
            continue
        value = _screen_seconds(row.get("collection_update_seconds"))
        if value is None:
            return None
        attempt = row.get("attempt_id")
        if attempt in seen:
            seconds.append(value)
        seen.add(attempt)
    return math.fsum(seconds)


def _screen_ancestry(directory: Path, details: Record) -> dict[str, Record]:
    """Read the active checkpoint chain through its existing host integrity owner.

    directory is one run and details is its immutable run_details.json. The
    latest pointer must stay inside checkpoints and match the description. Join
    the run ID and complete config before accepting its active actor boundaries.
    No numerical arrays are loaded, and historical source bytes are not required.
    """
    from marl_battlegrounds.training import checkpoints
    from marl_battlegrounds.training._run_io import checkpoint_ancestry

    pointer = _screen_object(directory / "latest_checkpoint.json")
    identifier = pointer.get("checkpoint_id")
    relative = pointer.get("relative_path")
    if (
        not isinstance(identifier, str)
        or not isinstance(relative, str)
        or Path(relative).parent != Path("checkpoints")
        or not (
            Path(relative).name == identifier
            or Path(relative).name.endswith(f"_{identifier}")
        )
        or Path(identifier).name != identifier
    ):
        raise ValueError("Screen run has no valid active checkpoint pointer")
    checkpoint = checkpoints.read_checkpoint_description(directory / relative)
    if checkpoint["checkpoint_id"] != identifier or any(
        checkpoint["metadata"].get(key) != details.get(key)
        for key in ("run_id", "config", "source", "dependencies", "execution")
    ):
        raise ValueError("Screen checkpoint differs from its recorded experiment")
    return checkpoint_ancestry(directory, checkpoint)


def _screen_evidence(
    result: Mapping[str, Any],
    *,
    run_id: str | None,
    seed: int | None,
    deployment: Mapping[str, Any] | None = None,
) -> tuple[Record, list[Record]]:
    """Verify one saved Random task and return its summary and original evaluation rows.

    result is a runner random_diagnostics entry, including original task and
    artifact identities. run_id and seed bind a fresh capture to its own run;
    both are None only for the explicitly shared original initialization.
    deployment is the owning run's fixed partner and roster declaration, or None
    for the historical bare actor. It is checked against the saved task.
    The result's own red_zone_depth (None when it was saved before the Red Zone
    rule) selects the task layout and whether rows carry recorded kills.
    Existing validation owns task and score verification;
    Evaluation owns CSV loading. No actor is applied and no game or file is changed.
    """
    from marl_battlegrounds.training.validation import (
        _verified_random_result,  # pyright: ignore[reportPrivateUsage]
    )

    return _verified_random_result(
        result,
        actor_digest=result["actor_digest"],
        seed_pairs=result["seed_pairs"],
        checkpoint_id=result["checkpoint_id"],
        env_steps=result["env_steps"],
        run_id=run_id,
        seed=seed,
        red_zone_depth=result.get("red_zone_depth"),
        deployment=deployment,
    )


def _screen_statistics(
    rows: Sequence[Record], result: Record, *, summary: Record | None = None
) -> Record:
    """Keep both spawn ends in each map/seed block for score and kill margin.

    Complete coverage and native scores are checked by summarize_validation.
    A supplied summary must be its verified result for these same rows; reuse it
    to avoid reducing scores twice after the saved-evidence integrity check.
    All points use the same private bootstrap seed, so matching blocks receive
    matching resamples across cases. Kills come from the columns _kill_columns
    names for this result (recorded kills when it carries a Red Zone depth,
    otherwise its scores). Optional kills must be finite and nonnegative when
    present; absent kills leave that diagnostic unavailable. Shared sampling
    evidence also applies to the kill interval. Unsupported bounds remain only
    explicitly conditional, while the descriptive mean remains available.
    """
    team_a, team_b = _kill_columns(result)
    if summary is None:
        summary = summarize_validation(
            rows,
            maps=result["maps"],
            opponents=tuple(member["name"] for member in result["members"])
            if "partners" in result
            else ("Random",),
            seed_pairs=result["seed_pairs"],
            actual_kills=(team_a, team_b) == _KILL_COLUMNS,
        )
    summary = _validation_sampling_interval(
        summary, result.get("sampling_evidence"), result["seed_pairs"]
    )
    if summary["score"] != result.get("score") or summary["cells"] != result.get(
        "cells"
    ):
        raise ValueError("Random summary differs from its saved game rows")
    blocks: dict[int, dict[int, tuple[float, float | None]]] = {}
    for map_id in result["maps"]:
        grouped: dict[int, list[Record]] = {}
        for row in rows:
            if row["map_id"] == map_id:
                grouped.setdefault(row["seed_id"], []).append(row)
        mapped: dict[int, tuple[float, float | None]] = {}
        for seed, pair in sorted(grouped.items()):
            margins: list[float] = []
            for row in pair:
                first, second = row.get(team_a), row.get(team_b)
                if any(
                    value is not None and _screen_seconds(value) is None
                    for value in (first, second)
                ):
                    raise ValueError(
                        "Random kill scores must be finite and nonnegative"
                    )
                if first is not None and second is not None:
                    margins.append(float(first) - float(second))
            mapped[seed] = (
                math.fsum(_score(row["system_game_score"]) for row in pair) / len(pair),
                math.fsum(margins) / len(pair) if len(margins) == len(pair) else None,
            )
        blocks[map_id] = mapped
    margin_blocks = [
        np.asarray([pair[1] for pair in mapped.values()], dtype=np.float64)
        for mapped in blocks.values()
    ]
    if all(np.isfinite(values).all() for values in margin_blocks):
        low, high = _bootstrap(margin_blocks, draws=2000, seed=19_044_001)
        margin: float | None = float(
            np.mean([values.mean() for values in margin_blocks])
        )
    else:
        margin, low, high = None, None, None
    conditional_bounds = {}
    if summary.get("interval_status", "Available") != "Available":
        conditional_bounds = {
            "conditional_kill_margin_ci_low": low,
            "conditional_kill_margin_ci_high": high,
        }
        low, high = None, None
    return {
        **summary,
        **conditional_bounds,
        **{
            name: (
                math.fsum(cell[column] for cell in summary["cells"])
                / len(summary["cells"])
                if all(cell.get(column) is not None for cell in summary["cells"])
                else None
            )
            for name, column in (
                ("mean_kills_for", f"mean_{team_a}"),
                ("mean_kills_against", f"mean_{team_b}"),
            )
        },
        **{
            name: sum(cell[name] for cell in summary["cells"])
            for name in ("wins", "draws", "losses")
        },
        "kill_margin": margin,
        "kill_margin_ci_low": low,
        "kill_margin_ci_high": high,
        "blocks": blocks,
    }


def _screen_change(current: Record, initial: Record) -> Record:
    """Compare one checkpoint to shared initialization using matching whole blocks.

    Both inputs are _screen_statistics results. Missing or changed map/seed
    coverage raises ValueError; no unpaired fallback is used. Return descriptive
    mean changes and intervals for fixed actors, never training-seed uncertainty.
    Either input's unsupported sampling suppresses the public change interval;
    it cannot regain support by subtracting initialization.
    """
    before, after = initial["blocks"], current["blocks"]
    if before.keys() != after.keys() or any(
        before[map_id].keys() != after[map_id].keys() for map_id in before
    ):
        raise ValueError("Random checks do not share the same map/seed blocks")
    result: Record = {}
    for index, name in enumerate(("score", "kill_margin")):
        values = [
            np.asarray(
                [
                    after[map_id][seed][index] - before[map_id][seed][index]
                    if after[map_id][seed][index] is not None
                    and before[map_id][seed][index] is not None
                    else np.nan
                    for seed in sorted(before[map_id])
                ],
                dtype=np.float64,
            )
            for map_id in before
        ]
        mean, low, high = None, None, None
        if all(np.isfinite(block).all() for block in values):
            mean = float(np.mean([block.mean() for block in values]))
            low, high = _bootstrap(values, draws=2000, seed=19_044_001)
        unsupported = [
            row["interval_status"]
            for row in (current, initial)
            if row.get("interval_status", "Available") != "Available"
        ]
        if unsupported:
            result[f"conditional_{name}_change_ci_low"] = low
            result[f"conditional_{name}_change_ci_high"] = high
            low, high = None, None
            result["change_interval_status"] = "; ".join(dict.fromkeys(unsupported))
        result[f"{name}_change_from_initial"] = mean
        result[f"{name}_change_ci_low"] = low
        result[f"{name}_change_ci_high"] = high
    return result


def _screen_task_identity(result: Mapping[str, Any]) -> str:
    """Describe immutable scientific fields, excluding case timing and reuse links."""
    excluded = {
        "elapsed_seconds",
        "training_seconds",
        "wall_seconds",
        "reference_path",
        "reused_initialization",
    }
    return json.dumps(
        {key: value for key, value in result.items() if key not in excluded},
        sort_keys=True,
    )


def _screen_reference(result: Record, package: Path) -> None:
    """Check that a reused initialization keeps its original task and actor IDs.

    reference_path names the copied original summary, absolute or package-relative.
    Its timing belongs to the first case and is deliberately excluded from this
    comparison. Reuse without a complete zero-step reference is rejected.
    """
    reused = result.get("reused_initialization")
    if type(reused) is not bool or reused != (result.get("reference_path") is not None):
        raise ValueError("Random initialization reuse needs its exact reference")
    if not reused:
        return
    value = result.get("reference_path")
    if not isinstance(value, str) or not value:
        raise ValueError("Reused initialization needs its original reference_path")
    path = Path(value)
    original = _screen_object(path if path.is_absolute() else package / path)
    if (
        result.get("env_steps") != 0
        or original.get("env_steps") != 0
        or original.get("complete") is not True
        or original.get("reused_initialization") is not False
        or original.get("reference_path") is not None
        or _screen_task_identity(original) != _screen_task_identity(result)
    ):
        raise ValueError(
            "Shared initialization changed its original scientific identity"
        )


def _screen_plot(points: Sequence[Record], destination: Path) -> dict[str, str]:
    """Plot combat improvement and task score against elapsed minutes and steps.

    Different recorded validation depths have separate curves, even before
    initialization completes. Missing x values are not invented. Each case
    references its shared initial
    task, but intervals were computed only once for each unique task. Write two
    derived images atomically; original evidence and numerical state stay intact.
    The first row uses paired kill-margin change since initialization, which can
    distinguish short-run combat progress even when every game ends in a draw.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    groups = tuple(
        dict.fromkeys((row["case"], row.get("red_zone_depth")) for row in points)
    )
    batches = sorted({row["num_envs"] for row in points})
    lengths = sorted({row["rollout_length"] for row in points})
    colors = plt.get_cmap("tab10")
    styles = ("-", "--", "-.", ":")
    handles: list[Any] = []
    labels: list[str] = []
    for name, depth in groups:
        rows = sorted(
            (
                row
                for row in points
                if row["case"] == name and row.get("red_zone_depth") == depth
            ),
            key=lambda row: row["env_steps"],
        )
        label = (
            f"{name} / Training Red Zone: "
            f"{_red_zone_label(rows[0].get('training_red_zone_depth'))}"
            f"\nValidation Red Zone: {_red_zone_label(rows[0].get('red_zone_depth'))}"
        )
        color = colors(batches.index(rows[0]["num_envs"]) % 10)
        style = styles[lengths.index(rows[0]["rollout_length"]) % len(styles)]
        for column, x_name in enumerate(
            ("wall_seconds", "warm_training_seconds", "env_steps")
        ):
            for index, metric in enumerate(
                ("kill_margin_change_from_initial", "score")
            ):
                selected = [
                    row
                    for row in rows
                    if row.get(x_name) is not None and row.get(metric) is not None
                ]
                if not selected:
                    continue
                divisor = 1 if x_name == "env_steps" else 60
                x = [row[x_name] / divisor for row in selected]
                low_key = "ci_low" if metric == "score" else "kill_margin_change_ci_low"
                high_key = (
                    "ci_high" if metric == "score" else "kill_margin_change_ci_high"
                )
                line = axes[index, column].plot(
                    x,
                    [row[metric] for row in selected],
                    marker="o",
                    markersize=4,
                    linewidth=1.4,
                    color=color,
                    linestyle=style,
                    label=label,
                )[0]
                axes[index, column].vlines(
                    x,
                    [row[low_key] for row in selected],
                    [row[high_key] for row in selected],
                    colors=line.get_color(),
                    alpha=0.3,
                )
                if label not in labels:
                    handles.append(line)
                    labels.append(label)
    for column, label in enumerate(
        (
            "Elapsed Run Time (Minutes, Including Overhead)",
            "Training Time After Each Attempt's First Block (Minutes)",
            "Environment Transitions",
        )
    ):
        for index, metric in enumerate(
            (
                "Change In Kills Minus Deaths Per Game\n"
                "Since Before Training (Higher Is Better)",
                "Game Result Score Against Random\nWin = 1, Draw = 0.5, Loss = 0",
            )
        ):
            axis = axes[index, column]
            axis.set_xlabel(label)
            axis.set_ylabel(metric)
            axis.grid(alpha=0.2)
            if not axis.lines:
                axis.text(
                    0.5,
                    0.5,
                    "No Completed Games Against Random",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="0.4",
                )
        axes[1, column].set_ylim(-0.02, 1.02)
        axes[0, column].axhline(0, color="0.5", linewidth=0.7, linestyle=":")
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    figure.suptitle(
        "Short MAPPO Screen: Combat Improvement For Time Spent\n"
        "Games Against Random; Error Bars Show Game Variation, "
        "Not Training-Seed Variation",
        fontsize=13,
    )
    paths: dict[str, str] = {}
    try:
        for suffix in ("png", "svg"):
            path = destination / f"learning_curves.{suffix}"
            temporary = destination / f".learning_curves.{suffix}.tmp"
            figure.savefig(temporary, format=suffix, dpi=150)
            os.replace(temporary, path)
            paths[suffix] = str(path)
    finally:
        plt.close(figure)
    return paths


def analyze_screen(
    package_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    render_plots: bool = True,
) -> Record:
    """Report every declared configuration from a complete or unfinished screen.

    Parameters
    ----------
    package_dir : path
        Frozen screen package with declaration.json and its ordered cases list. Optional
        study.json and budgets.json hold lifecycle facts and fixed timing-based budgets.
        Trials use jobs/<name>/run and ordinary trainer records, including
        random_diagnostics.json. Referenced shared initial results retain their original
        task/checkpoint/actor IDs and raw evaluation paths. Initial actors are shared
        only within one training seed. Started cases keep their frozen config, latest
        checkpoint pointer and ancestor descriptions. Original actor and evaluation
        evidence remain readable.
    output_dir : path or None, default=None
        Destination for derived reports, default package_dir/reports. Replace only
        run_summary.md, baseline_trials.csv, validation_cells.csv,
        learning_curve.csv and learning_curves.png/.svg.
    render_plots : bool, default=True
        Draw the ordinary screen plots. False skips plotting and its imports;
        callers making a larger study report can reuse the verified tables.

    Returns
    -------
    dict
        artifacts maps summary/trials/cells/curve/png/svg to paths. cases includes
        failed and unstarted trials. unique_validation_tasks counts shared checks
        once. evidence_errors names unreadable or inconsistent case evidence.
        complete requires every raw run and expected Random capture to be complete;
        it does not depend on the driver's terminal status or select a winner.

    Raises
    ------
    ValueError, OSError
        The declaration is missing, malformed or repeats case names, or reports
        cannot be written. Invalid individual runs remain visible as evidence
        errors, allowing valid cases and unfinished studies to be reported.
    ImportError
        Plotting dependencies from the viz extra are unavailable.

    Notes
    -----
    Host-only, headless analysis; no learner or game runs. Shared map/seed blocks
    keep both spawn ends together. Changes from initialization use matching blocks.
    Intervals describe game sampling for fixed actors, not training-seed variation.
    Combat change against Random over wall time is the short-screen comparison;
    native outcomes remain separate. Kills and deaths are recorded kills for
    results that carry a Red Zone depth and the native scores for older
    results, where they were equal (_kill_columns). All draws do not
    establish absent learning.
    The secondary training axis excludes each attempt's
    first complete block, including its useful work; that block is not pure compile
    time. Missing timings remain missing. Original evidence is never changed.
    """
    package = Path(package_dir).resolve()
    if type(render_plots) is not bool:
        raise TypeError("render_plots must be bool")
    declaration = _screen_object(package / "declaration.json")
    raw_cases = declaration.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Screen declaration needs a nonempty ordered cases list")
    declared: list[Record] = []
    names: set[str] = set()
    for raw in cast(list[object], raw_cases):
        if not isinstance(raw, dict):
            raise ValueError("Each declared screen case must be an object")
        row = cast(Record, raw)
        name = row.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or Path(name).name != name
            or name in (".", "..")
        ):
            raise ValueError("Screen case names must be distinct directory names")
        names.add(name)
        declared.append(row)
    study = _screen_object(package / "study.json")
    budgets = _screen_object(package / "budgets.json")
    budget_rows = budgets.get("cases", [])
    if not isinstance(budget_rows, list):
        raise ValueError("Screen budgets need a cases list")
    by_budget: dict[str, Record] = {}
    for value in cast(list[object], budget_rows):
        if not isinstance(value, dict):
            raise ValueError("Each screen budget must be an object")
        row = cast(Record, value)
        if row.get("name") not in names or row["name"] in by_budget:
            raise ValueError("Screen budgets have an unknown or repeated case")
        by_budget[row["name"]] = row
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else package / "reports"
    )
    destination.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, Record] = {}
    shared = _screen_object(package / "shared" / "initialization.json")
    initial_identities: dict[int, tuple[object, object]] = {}
    cases: list[Record] = []
    curves: list[Record] = []
    points: list[Record] = []
    errors: list[Record] = []
    for definition in declared:
        name = definition["name"]
        budget = by_budget.get(name, {})
        lifecycle = study.get("cases", {}).get(name, {})
        directory = package / "jobs" / name / "run"
        case: Record = {
            "case": name,
            "num_envs": definition.get("num_envs"),
            "rollout_length": definition.get("rollout_length"),
            "status": lifecycle.get("status", "unstarted"),
            "phase": lifecycle.get("phase"),
            "error": lifecycle.get("error"),
            "run_dir": str(directory),
            "declared_env_steps": budget.get("total_env_steps"),
            "declared_updates": budget.get("updates"),
            "calibration_median_update_seconds": budget.get("median_update_seconds"),
            "complete": False,
            "red_zone_depth": None,
        }
        cases.append(case)
        if not (directory / "run_details.json").exists():
            if case["status"] == "complete":
                errors.append(
                    {"case": name, "error": "Complete case has no run_details.json"}
                )
            continue
        try:
            summary = _summary(directory)
            case.update(
                {
                    key: summary.get(key)
                    for key in (
                        "run_id",
                        "status",
                        "env_steps",
                        "wall_seconds",
                        "elapsed_seconds",
                        "phase_costs",
                        "memory",
                        "exposure",
                        "recorded_failures",
                        "incomplete_events",
                        "final_actor",
                        "seed",
                        "details_path",
                        "red_zone_depth",
                    )
                }
            )
            if summary["status"] == "recovering":
                continue
            details = _screen_object(directory / "run_details.json")
            config = cast(Record, details.get("config", {}))
            training_seed = _integer(config.get("seed"), "training seed")
            if definition.get("seed", training_seed) != training_seed:
                raise ValueError("Saved training seed differs from its declaration")
            declared_config = budget.get("config")
            if declared_config is None and budget.get("config_path"):
                declared_config = _screen_object(package / budget["config_path"])
            if declared_config is not None and config != declared_config:
                raise ValueError("Saved config differs from its frozen case config")
            if budget and declared_config is None:
                raise ValueError("Screen budget has no frozen case config")
            if (
                config.get("num_envs") != definition.get("num_envs")
                or config.get("ppo", {}).get("rollout_length")
                != definition.get("rollout_length")
                or (
                    budget.get("total_env_steps") is not None
                    and config.get("total_env_steps") != budget["total_env_steps"]
                )
            ):
                raise ValueError("Saved run differs from its declared shape or budget")
            ancestry = _screen_ancestry(directory, details)
            exposure = summary.get("exposure")
            if isinstance(exposure, dict):
                case["exposure"] = {
                    key: exposure.get(key)
                    for key in (
                        "starts_by_episode_stage",
                        "steps_by_episode_stage",
                        "actor_decisions_by_episode_stage",
                        "steps_by_map",
                        "starts_by_opponent",
                        "steps_by_opponent",
                        "stage_completed",
                        "stage_k20_count",
                        "incomplete_games",
                    )
                }
            updates = _read_rows(directory / "training_updates.jsonl")
            latest = updates[-1] if updates else {}
            case.update(
                {
                    key: latest.get(key)
                    for key in (
                        "training_seconds",
                        "actor_decisions",
                        "used_policy_samples",
                        "used_value_samples",
                    )
                }
            )
            case["first_block_seconds"] = (
                updates[0].get("collection_update_seconds") if updates else None
            )
            case["warm_training_seconds"] = _screen_warm_seconds(
                updates, latest.get("env_steps", 0)
            )
            for row in updates:
                curves.append(
                    {
                        "case": name,
                        "kind": "update",
                        "red_zone_depth": summary["red_zone_depth"],
                        "training_red_zone_depth": summary["red_zone_depth"],
                        **{
                            key: value
                            for key, value in row.items()
                            if not isinstance(value, (dict, list))
                        },
                    }
                )
            raw_results = _read_json(directory / "random_diagnostics.json", [])
            if not isinstance(raw_results, list):
                raise ValueError("random_diagnostics.json must contain a list")
            seen_steps: set[int] = set()
            local: list[Record] = []
            for raw in cast(list[object], raw_results):
                if not isinstance(raw, dict):
                    raise ValueError("Random diagnostic entries must be objects")
                result = cast(Record, raw)
                if result.get("seed_pairs") != config.get(
                    "random_diagnostic_seed_pairs"
                ):
                    raise ValueError(
                        "Random seed-pair count differs from the run config"
                    )
                steps = _integer(result.get("env_steps"), "diagnostic env_steps")
                if steps in seen_steps:
                    raise ValueError("A case repeats a Random diagnostic step")
                seen_steps.add(steps)
                if result.get("complete") is not True:
                    continue
                _screen_reference(result, package)
                from marl_battlegrounds.training.checkpoints import (
                    _inference_digest,  # pyright: ignore[reportPrivateUsage]
                )

                reused = result["reused_initialization"]
                reference = config.get("random_initialization_result")
                expected_reuse = steps == 0 and reference is not None
                expected_reference = (
                    str(Path(cast(str, reference)).absolute())
                    if expected_reuse
                    else None
                )
                if (
                    reused != expected_reuse
                    or result.get("reference_path") != expected_reference
                ):
                    raise ValueError(
                        "Random initialization reuse differs from its declared config"
                    )
                origin = result.get("checkpoint_id")
                if not isinstance(origin, str):
                    raise ValueError("Random result needs its originating checkpoint")
                boundary = (
                    next(
                        (
                            item
                            for item in ancestry.values()
                            if item["counters"]["env_steps"] == 0
                        ),
                        None,
                    )
                    if reused
                    else ancestry.get(origin)
                )
                if (
                    boundary is None
                    or boundary["counters"]["env_steps"] != steps
                    or _inference_digest(boundary) != result.get("actor_digest")
                ):
                    raise ValueError(
                        "Random actor is outside this run's active checkpoint chain"
                    )
                task_id = result.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError("Random diagnostic needs its original task_id")
                if steps == 0:
                    identity_pair = (task_id, result.get("actor_digest"))
                    expected_initial = initial_identities.setdefault(
                        training_seed,
                        (shared["task_id"], shared.get("actor_digest"))
                        if shared.get("task_id") is not None
                        else identity_pair,
                    )
                    if expected_initial != identity_pair:
                        raise ValueError(
                            "Each training seed must share one original initial "
                            "task and actor"
                        )
                identity = _screen_task_identity(result)
                if task_id in tasks:
                    if tasks[task_id]["identity"] != identity:
                        raise ValueError(
                            "A shared task ID has conflicting scientific evidence"
                        )
                else:
                    verified, rows = _screen_evidence(
                        result,
                        run_id=None if reused else details["run_id"],
                        seed=None if reused else config["seed"],
                        deployment=details.get("validation_deployment"),
                    )
                    statistics = _screen_statistics(rows, result, summary=verified)
                    tasks[task_id] = {
                        "identity": identity,
                        "result": result,
                        "statistics": statistics,
                        "cases": [],
                    }
                task = tasks[task_id]
                if name not in task["cases"]:
                    task["cases"].append(name)
                point: Record = {
                    "case": name,
                    "kind": "random",
                    "red_zone_depth": result.get("red_zone_depth"),
                    "training_red_zone_depth": summary["red_zone_depth"],
                    "num_envs": definition["num_envs"],
                    "rollout_length": definition["rollout_length"],
                    "env_steps": steps,
                    "task_id": task_id,
                    "checkpoint_id": result.get("checkpoint_id"),
                    "actor_digest": result.get("actor_digest"),
                    "actor_path": result.get("actor_path"),
                    "summary_path": result.get("summary_path"),
                    "reference_path": result.get("reference_path"),
                    "reused_initialization": bool(result.get("reused_initialization")),
                    "wall_seconds": _screen_seconds(result.get("wall_seconds")),
                    "elapsed_seconds": _screen_seconds(result.get("elapsed_seconds")),
                    "training_seconds": _screen_seconds(result.get("training_seconds")),
                    "warm_training_seconds": _screen_warm_seconds(updates, steps),
                    **{
                        key: value
                        for key, value in task["statistics"].items()
                        if key not in ("blocks", "cells")
                    },
                }
                local.append(point)
            initial = next((row for row in local if row["env_steps"] == 0), None)
            if initial is not None:
                reference = tasks[initial["task_id"]]["statistics"]
                for point in local:
                    if point["red_zone_depth"] != initial["red_zone_depth"]:
                        raise ValueError(
                            "Random diagnostic depth differs from initialization; "
                            "paired changes need the same recorded rules"
                        )
                    point.update(
                        _screen_change(tasks[point["task_id"]]["statistics"], reference)
                    )
                    point["initial_task_id"] = initial["task_id"]
            points.extend(local)
            curves.extend(local)
            total = _integer(
                budget.get("total_env_steps", config.get("total_env_steps")),
                "declared total_env_steps",
                minimum=1,
            )
            expected = {0, total}
            expected.update(
                _integer(value, "diagnostic checkpoint_env_steps", minimum=1)
                for value in config.get(
                    "checkpoint_env_steps", budget.get("checkpoint_env_steps", [])
                )
            )
            case["validation_red_zone_depths"] = list(
                dict.fromkeys(row["red_zone_depth"] for row in local)
            )
            completed = {row["env_steps"] for row in local}
            case["expected_diagnostic_steps"] = sorted(expected)
            case["completed_diagnostic_steps"] = sorted(completed)
            final = next((row for row in local if row["env_steps"] == total), None)
            if final is not None:
                case["final_validation_red_zone_depth"] = final["red_zone_depth"]
                case.update(
                    {
                        f"final_{key}": final.get(key)
                        for key in (
                            "score",
                            "ci_low",
                            "ci_high",
                            "kill_margin",
                            "kill_margin_ci_low",
                            "kill_margin_ci_high",
                            "mean_kills_for",
                            "mean_kills_against",
                            "wins",
                            "draws",
                            "losses",
                            "score_change_from_initial",
                            "score_change_ci_low",
                            "score_change_ci_high",
                            "kill_margin_change_from_initial",
                            "kill_margin_change_ci_low",
                            "kill_margin_change_ci_high",
                            "interval_status",
                            "change_interval_status",
                            "task_id",
                        )
                    }
                )
            case["complete"] = (
                case["status"] == "complete"
                and case["env_steps"] == total
                and expected == completed
            )
        except (ValueError, OSError, KeyError, TypeError) as error:
            case["evidence_error"] = str(error)
            case["complete"] = False
            errors.append({"case": name, "error": str(error)})
    cells: list[Record] = []
    for task_id, task in tasks.items():
        result = task["result"]
        # Kills, not points: a Red Zone death gives 2 points but one kill.
        kills_for, kills_against = (f"mean_{name}" for name in _kill_columns(result))
        for cell in task["statistics"]["cells"]:
            cells.append(
                {
                    "task_id": task_id,
                    "cases": json.dumps(task["cases"]),
                    "checkpoint_id": result.get("checkpoint_id"),
                    "actor_digest": result.get("actor_digest"),
                    "env_steps": result["env_steps"],
                    "summary_path": result.get("summary_path"),
                    **cell,
                    "red_zone_depth": result.get("red_zone_depth"),
                    "mean_kill_margin": (
                        cell[kills_for] - cell[kills_against]
                        if cell.get(kills_for) is not None
                        and cell.get(kills_against) is not None
                        else None
                    ),
                }
            )
    artifacts = _screen_plot(points, destination) if render_plots else {}
    for key, filename, rows in (
        ("trials", "baseline_trials.csv", cases),
        ("cells", "validation_cells.csv", cells),
        ("curve", "learning_curve.csv", curves),
    ):
        path = destination / filename
        scalar_rows = [
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            for row in rows
        ]
        empty_columns = {
            "trials": "case,status,complete\n",
            "cells": "task_id,cases,map_id,score,mean_kill_margin\n",
            "curve": "case,kind,env_steps,wall_seconds,score,kill_margin\n",
        }
        _atomic_text(path, _csv(scalar_rows) if scalar_rows else empty_columns[key])
        artifacts[key] = str(path)
    complete = (
        len(by_budget) == len(cases)
        and all(row["complete"] for row in cases)
        and not errors
    )
    lines = [
        "# Configuration Screen Summary",
        "",
        f"Study Status: {study.get('status', 'Not Started')}. "
        f"Phase: {study.get('phase', 'Unavailable')}.",
        f"Complete Case Evidence: {sum(row['complete'] for row in cases)}"
        f"/{len(cases)}.",
        f"Total Study Time: {_screen_duration(study.get('elapsed_seconds'))}.",
        f"Unique Random Tasks: {len(tasks)}. Unique Evaluated Games: "
        f"{sum(task['statistics']['games'] for task in tasks.values())}.",
        "",
        "Frozen recipe, seed, order and grid: "
        f"[declaration.json]({package / 'declaration.json'}). "
        "Timing-based budgets and forecast: "
        f"[budgets.json]({package / 'budgets.json'}). "
        f"Original lifecycle: [study.json]({package / 'study.json'}).",
        "",
        "This short screen compares combat improvement against Random for the "
        "time spent. Wins are not required. We compare the change in average "
        "kills minus deaths per game since before training; higher is better. "
        "Kills and deaths are also shown separately. Winning the original task "
        "remains a separate outcome (win 1, draw 0.5, loss 0). "
        "No winner or next trial is selected automatically.",
        "Elapsed run time is the primary plot axis, in minutes. It includes "
        "setup, previous checks, saving and any pauses. Training time and environment "
        "transitions are secondary. The training axis excludes each attempt's "
        "first block, including its useful work; the excluded duration is not "
        "pure compilation time. Missing measurements stay unknown.",
        "Each capture's wall clock is saved just before its Random check. Final "
        "run and study totals also include that check and report generation.",
        "Intervals are descriptive game-sampling intervals for fixed actors. "
        "Whole map/seed blocks keep both spawn ends together, using matching "
        "resamples across cases. Changes from initialization use those same "
        "blocks. A shared initialization is counted once; case curves only "
        "reference it. These intervals do not measure training variation. "
        "Compare separate training seeds in the study's configuration summary; "
        "a short development screen alone does not establish a final best setting.",
        "The first case pays for the shared initialization evaluation in its wall "
        "time. Later cases reuse that evidence; their evaluation costs are lower "
        "by that shared work. The separate study clock includes the full cost.",
        "Different budgets change self-play snapshot timing. Equal-experience "
        "checkpoints compare these complete procedures. All draws mean that "
        "win/draw/loss outcomes do not distinguish these actors. They do not "
        "show that nothing was learned: kills and deaths can still improve. "
        "An interval collapsed to one number does not prove certainty beyond "
        "the games played. Combat progress against Random does not establish "
        "competence against stronger opponents.",
        "",
        "## Cases",
        "",
        "| Configuration | Training Red Zone Depth | Final Validation Red Zone Depth | "
        "State | Steps Completed | "
        "Elapsed Run Time | "
        "Kills / Deaths Per Game | Change In Kills Minus Deaths | "
        "Wins / Draws / Losses |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for case in cases:

        def shown(key: str, record: Record = case) -> str:
            """Format one optional saved case value; missing facts stay unavailable."""
            value = record.get(key)
            if value is None:
                return "Unknown"
            return f"{value:.2f}" if isinstance(value, float) else str(value)

        lines.append(
            f"| {case['case']} | {_red_zone_label(case['red_zone_depth'])} | "
            f"{_red_zone_label(case.get('final_validation_red_zone_depth'))} | "
            f"{shown('status')} | {shown('env_steps')} | "
            f"{_screen_duration(case.get('wall_seconds'))} | "
            f"{shown('final_mean_kills_for')} / {shown('final_mean_kills_against')} | "
            f"{shown('final_kill_margin_change_from_initial')} | "
            f"{shown('final_wins')} / {shown('final_draws')} / "
            f"{shown('final_losses')} |"
        )
    for case in cases:
        lines.extend(
            [
                "",
                f"### {case['case']}",
                "",
                f"Original run: [{case['run_dir']}]({case['run_dir']}).",
                "Recorded Validation Red Zone Depths: "
                + (
                    "; ".join(
                        _red_zone_label(depth)
                        for depth in case.get("validation_red_zone_depths", [])
                    )
                    or "No recorded validation"
                )
                + ".",
            ]
        )
        change = case.get("final_kill_margin_change_from_initial")
        if change is not None:
            low = case.get("final_kill_margin_change_ci_low")
            high = case.get("final_kill_margin_change_ci_high")
            interval = (
                f"{low:+.2f} to {high:+.2f}"
                if low is not None and high is not None
                else case.get("final_change_interval_status") or "Unavailable"
            )
            lines.append(
                f"Change In Kills Minus Deaths: {change:+.2f} per game since "
                f"before training. Paired Game Interval: {interval}. "
                "An available range reflects sampled games, not training seeds."
            )
        if case.get("error") or case.get("evidence_error"):
            lines.append(
                "Saved Failure Or Evidence Error: "
                f"{case.get('evidence_error') or case.get('error')}."
            )
        if "phase_costs" in case:
            lines.append(
                "Measured phase-owner costs across attempts (seconds): "
                + ", ".join(
                    f"{name}: {value['seconds']}"
                    if value["seconds"] is not None
                    else f"{name}: Unavailable"
                    for name, value in case["phase_costs"].items()
                )
                + ". Interrupted work can be missing from these measurements."
            )
        if case.get("memory") is not None:
            lines.extend(_memory_lines(case["memory"], label="Latest Saved Memory"))
        if case.get("exposure") is not None:
            lines.append(
                "Actual map, opponent and actor exposure: "
                f"[exposure.json]({Path(case['run_dir']) / 'exposure.json'})."
            )
        if case.get("final_actor"):
            lines.append(
                f"Final actor: [{case['final_actor']}]({case['final_actor']})."
            )
    if errors:
        lines.extend(["", "## Evidence Errors", ""])
        lines.extend(f"- {row['case']}: {row['error']}" for row in errors)
    path = destination / "run_summary.md"
    _atomic_text(path, "\n".join(lines) + "\n")
    artifacts["summary"] = str(path)
    return {
        "artifacts": artifacts,
        "cases": cases,
        "complete": complete,
        "unique_validation_tasks": len(tasks),
        "evidence_errors": errors,
        "qualification": (
            "Descriptive one-seed configuration screen; "
            "no automatic winner or learning qualification"
        ),
    }
