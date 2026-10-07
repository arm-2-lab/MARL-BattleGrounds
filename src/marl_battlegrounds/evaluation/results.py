"""Read scoped evaluation tables without running games or changing saved files.

Results keep the older convenience fields and add bounded NumPy table access.
Saved views use one committed manifest snapshot. They never recover a writer,
load JAX, reconstruct optional measurements or fit tournament ratings.
Head-to-head views count saved outcomes and points; their matrix is a plain pivot.
``system_metrics`` shows every System's metric columns from its own side (each
System as Team A, through ``metric_catalog.mirror_metric``), combines games by
each column's ``summary_weight`` and resamples whole spawn pairs for intervals.
``execution_settings`` reports whether deterministic GPU mode was declared.
``table("policy_assignments")`` and the member columns of
``system_metrics(by="game")`` say which member of a pool or team played each
slot of each saved game (``_member_labels``, ``_game_members``).
Shared private helpers own the side of each game (``_directions``), each
scheduled game's sampling facts (``_game_plan``, ``_plan_record``), and the draw
order and interval rule that ``point_margin_summary`` and ``system_metrics``
share (``_bootstrap_indices``, ``_interval_rule``: unknown sampling alone keeps
the range, labeled "Conditional: Sampling not verified", the label Elo rankings
give unknown sampling).
Whole-table allocation is explicit through ``table``; ``iter_table`` bounds raw
row batches.
Each saved run reads with the full-report header of its own recorded scalar
schema (14 before the Red Zone columns, 15 now).
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    FULL_METRIC_NAMES_BY_SCHEMA_VERSION,
    METRIC_COLUMNS_BY_NAME,
    METRIC_SCHEMA_ID,
    METRIC_SCHEMA_VERSION,
    PRIORITY_METRIC_NAMES,
    mirror_metric,
)
from marl_battlegrounds.evaluation.run_writer import (
    ASSIGNMENT_COLUMNS,
    EPISODE_COLUMNS,
    IDENTITY_COLUMNS,
    MATCH_COLUMNS,
)
from marl_battlegrounds.evaluation.sampling_evidence import (
    CONDITIONAL_INTERVAL_ASSUMPTION,
    CONDITIONAL_INTERVAL_STATUS,
)
from marl_battlegrounds.evaluation.scalar_reports import (
    has_members,
    iter_policy_assignments,
    iter_scalar_rows,
    iter_summary_rows,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.replay_v4 import ReplayArtifactV4
    from marl_battlegrounds.evaluation.tournament_records import TournamentRecords

Columns = dict[str, NDArray[np.generic]]
Row = dict[str, Any]
_TABLES = (
    "episodes",
    "priority_metrics",
    "full_metrics",
    "matches",
    "tournament_results",
    "tournament_rankings",
    "matchup_results",
    "map_results",
    "tournament_headline_metrics",
    "policy_assignments",
)
_SUMMARIES = frozenset(
    {
        "tournament_results",
        "tournament_rankings",
        "matchup_results",
        "map_results",
        "tournament_headline_metrics",
    }
)
# Member columns that system_metrics(by="game") adds, from the System's side.
_MEMBER_COLUMNS = (
    *(f"agent_{slot}_member" for slot in range(10)),
    "member",
    "opponent_member",
)
_IN_MEMORY_MEMBERS = "Members are recorded in saved runs; pass output_dir"
_NO_MEMBERS = "No System in this scope has members"
_FILES = {name: f"{name}.csv" for name in _TABLES if name != "tournament_rankings"}
_FILES["matches"] = "match_results.csv"
_TEXT = frozenset(
    {
        "run_id",
        "phase",
        "pass_id",
        "config_id",
        "team_a_policy",
        "team_b_policy",
        "checkpoint_id",
        "bootstrap_group",
        "system_id",
        "system_name",
        "opponent_id",
        "opponent_name",
        "spawn_end",
        "played_as",
        "policy",
        "opponent",
        "determinism",
        "conditional_interval_assumption",
        "sampling_unit",
        "unavailable_reason",
        "team",
        *_MEMBER_COLUMNS,
    }
)
_INTS = frozenset(
    {
        "global_slot",
        "policy_id",
        "decision_start",
        "decision_stop",
        "episode_id",
        "seed_id",
        "map_id",
        "block_id",
        "outcome",
        "rank",
        "matches",
        "independent_blocks",
        "declared_blocks",
        "scheduled_games",
        "completed_games",
        "supported_independent_sampling_units",
        "games_played",
        "games",
        "completed_pairs",
        "independent_units",
        "bootstrap_draws",
        "bootstrap_seed",
        "total_steps_played",
        "episode_length_min",
        "episode_length_max",
        "score_difference_min",
        "score_difference_max",
        "total_score_difference",
        "wins",
        "draws",
        "losses",
        *(
            f"agent_{slot}_{suffix}"
            for slot in range(10)
            for suffix in ("class_id", "active")
        ),
    }
)
_EXACT = frozenset({"episode_length", "team_a_score", "team_b_score"})
_NULLABLE = frozenset({"seed_id", "map_id", "pass_id", "block_id", "rank"})
_HEADERS: dict[str, tuple[str, ...]] = {
    "episodes.csv": EPISODE_COLUMNS,
    "priority_metrics.csv": (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES),
    "full_metrics.csv": (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES),
    "match_results.csv": MATCH_COLUMNS,
    "policy_assignments.csv": ASSIGNMENT_COLUMNS,
}
_CELL_TYPES = {
    "text": ("text", "None when absent"),
    "integer": ("integer", "None when absent; never rounded through float"),
    "float32": ("float32", "NaN when a measurement is unavailable"),
    "float64": ("float64", "NaN when a measurement is unavailable"),
    "nullable_float64": ("float64", "None when the derived value is unavailable"),
}


def _cell_type(column: str, table: str, current: bool) -> tuple[str, str]:
    """Share immutable type/null descriptions instead of thousands of tiny dicts."""
    if column in _TEXT or column.endswith("_status"):
        return _CELL_TYPES["text"]
    if column in _INTS or (
        column in _EXACT and table in {"episodes", "matches"} and current
    ):
        return _CELL_TYPES["integer"]
    if column in {"kd_ratio", "system_game_score"}:
        return _CELL_TYPES["nullable_float64"]
    return _CELL_TYPES[
        "float32"
        if current and table in {"priority_metrics", "full_metrics", "matches"}
        else "float64"
    ]


def _mapping(value: object) -> dict[str, Any]:
    """Return a plain host mapping or an empty mapping for absent metadata."""
    return dict(cast(Mapping[str, Any], value)) if isinstance(value, Mapping) else {}


def _stamp(path: Path) -> tuple[int, int, int, int]:
    """Keep nanosecond file identity; stat-result equality loses subsecond edits."""
    value = path.stat()
    return value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _schema(manifest: Mapping[str, Any]) -> None:
    """Reject unsupported host/scalar pairs before opening any table.

    Host schema 1 accepts scalar schemas 1 to 13. Host schema 2 accepts each
    scalar schema that FULL_METRIC_NAMES_BY_SCHEMA_VERSION lists (14 and 15).
    Raises ValueError for any other pair, a different metric schema ID or a
    pending recording restore.
    """
    host, scalar = manifest.get("schema_version"), manifest.get("metric_schema_version")
    if manifest.get("metric_schema_id") != METRIC_SCHEMA_ID or not (
        type(host) is int
        and type(scalar) is int
        and (
            (host == 2 and scalar in FULL_METRIC_NAMES_BY_SCHEMA_VERSION)
            or (host == 1 and 1 <= scalar <= 13)
        )
    ):
        raise ValueError("unsupported result host/scalar schema pair")
    if "recording_restore" in manifest:
        raise ValueError(
            "recording restore is pending; complete the explicit restore first"
        )


def _headers(manifest: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return the raw table headers that a checked manifest's version expects.

    A host-schema-2 manifest gets the full-report header of its own scalar
    schema from FULL_METRIC_NAMES_BY_SCHEMA_VERSION, so a saved schema-14 table
    reads with its 11,148 original columns. Host schema 1 keeps the current
    headers; its tables are always read by their stored headers. Call only
    after _schema has accepted the manifest.
    """
    if manifest["schema_version"] != 2:
        return _HEADERS
    names = FULL_METRIC_NAMES_BY_SCHEMA_VERSION[manifest["metric_schema_version"]]
    return {**_HEADERS, "full_metrics.csv": (*IDENTITY_COLUMNS, *names)}


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    """Read one exact run, refusing parent-directory guessing and pending restore."""
    path = run_dir / "run_details.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("run_dir must name one saved run containing run_details.json")
    try:
        manifest = json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as error:
        raise ValueError("run_details.json is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("run_details.json must contain a mapping")
    _schema(cast(dict[str, Any], manifest))
    return cast(dict[str, Any], manifest)


def _select_passes(
    manifest: Mapping[str, Any], phase: str | None, pass_id: str | None
) -> dict[str, Row]:
    """Resolve exact recorded scope, rejecting ambiguous or empty selections."""
    for value in (phase, pass_id):
        if value is not None and (
            not isinstance(cast(object, value), str) or not value
        ):
            raise ValueError("phase and pass_id must be nonempty strings when supplied")
    selected = {
        key: _mapping(entry)
        for key, entry in _mapping(manifest.get("passes")).items()
        if (phase is None or _mapping(entry).get("phase") == phase)
        and (pass_id is None or _mapping(entry).get("pass_id") == pass_id)
    }
    if not selected:
        raise ValueError("result selection contains no recorded passes")
    if phase is None and pass_id is not None and len(selected) != 1:
        raise ValueError("pass_id appears in several phases; supply phase too")
    identities = [
        (entry.get("phase"), entry.get("pass_id")) for entry in selected.values()
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("run manifest repeats a pass identity")
    return selected


def _scope(selected: Mapping[str, Row], whole_tournament: bool) -> Row:
    """Describe selected recorded passes without inventing a combined pass."""
    return {
        "passes": tuple(
            (entry["phase"], entry["pass_id"]) for entry in selected.values()
        ),
        "pass_keys": tuple(selected),
        "whole_tournament": whole_tournament,
    }


def _historical_coordinator(entry: Row, manifest: Mapping[str, Any]) -> bool:
    """Recognize the actual old zero-game coordinator from its complete contract."""
    if (
        entry.get("phase") != "tournament"
        or entry.get("pass_id") != "schedule"
        or entry.get("episodes")
    ):
        return False
    details = _mapping(manifest.get("details"))
    expected = details.get("num_matches")
    tables = _mapping(manifest.get("tables"))
    passes = [
        value
        for value in _mapping(manifest.get("passes")).values()
        if value.get("phase") == "tournament" and value.get("pass_id") != "schedule"
    ]
    identifiers = [
        value for item in passes for value in item.get("completed_episode_ids", ())
    ]
    return bool(
        _mapping(manifest.get("tournament_summary")).get("digest")
        and details.get("schedule_digest")
        and details.get("policies")
        and details.get("map_ids")
        and details.get("configuration_ids_by_map")
        and type(expected) is int
        and expected > 0
        and len(identifiers) == len(set(identifiers)) == expected
        and all(
            set(map(str, item.get("completed_episode_ids", ())))
            == set(item.get("episodes", {}))
            for item in passes
        )
        and tables.get("match_results.csv", {}).get("rows") == expected
        and all(
            tables.get(name, {}).get("rows", 0) > 0
            for name in (
                "tournament_results.csv",
                "matchup_results.csv",
                "map_results.csv",
            )
        )
    )


def _state(entry: Row, manifest: Mapping[str, Any]) -> tuple[str, str | None]:
    """Read explicit status or prove only the older recorded completion contract."""
    marker = _mapping(entry.get("result_state"))
    if marker:
        if marker.get("version") != 1 or marker.get("status") not in {
            "complete",
            "incomplete",
            "failed",
        }:
            raise ValueError("invalid pass result_state")
        return str(marker["status"]), marker.get("reason")
    if entry.get("pass_role") == "tournament_coordinator":
        summary = _mapping(manifest.get("tournament_summary"))
        qualification = _mapping(summary.get("qualification"))
        return (
            ("complete", None)
            if qualification.get("status") == "complete"
            else ("incomplete", "Tournament summary is not complete")
        )
    schedule = _mapping(entry.get("episodes"))
    completed = entry.get("completed_episode_ids", ())
    if schedule and set(map(str, completed)) == set(schedule):
        details = _mapping(entry.get("details"))
        replays = _mapping(entry.get("replays"))
        if all(str(value) in replays for value in details.get("replay_episodes", ())):
            return "complete", None
    # Older tournaments had a zero-game coordinator with a documented summary.
    if _historical_coordinator(entry, manifest):
        return "complete", None
    return "incomplete", "No complete recorded schedule and coverage proof"


def _source_coverage(
    entry: Mapping[str, Any],
    lengths: Mapping[tuple[str, str, str], int],
) -> tuple[Row, ...]:
    """Count saved source choices, including choices with no scheduled game.

    Repeated choices share one count and retain all their original positions.
    Counts join each recorded game by its declared map and source identity, never
    today's catalog. Zero played games means zero steps; a missing played length
    means None. Older or exact schedules without a source-choice list return ().
    """
    contract = _mapping(_mapping(entry.get("details")).get("evaluation_contract"))
    groups: dict[tuple[object, object], Row] = {}
    for index, choice in enumerate(contract.get("source_choices", ())):
        identity = choice.get("map_id"), choice.get("source_config_id")
        if identity not in groups:
            groups[identity] = {
                "map_id": identity[0],
                "source_config_id": identity[1],
                "source_choice_indices": [],
                "map_metadata": choice.get("map_metadata"),
                "scheduled_games": 0,
                "completed_games": 0,
                "completed_steps": 0,
            }
        groups[identity]["source_choice_indices"].append(index)
    completed = set(map(str, entry.get("completed_episode_ids", ())))
    for identifier, declaration in _mapping(entry.get("episodes")).items():
        group = groups.get(
            (declaration.get("map_id"), declaration.get("source_config_id"))
        )
        if group is None:
            continue
        group["scheduled_games"] += 1
        if identifier in completed:
            group["completed_games"] += 1
            length = lengths.get((entry["phase"], entry["pass_id"], identifier))
            if length is None:
                group["completed_steps"] = None
            elif group["completed_steps"] is not None:
                group["completed_steps"] += length
    for group in groups.values():
        group["source_choice_indices"] = tuple(group["source_choice_indices"])
    return tuple(groups.values())


def _columns(rows: Sequence[Row], *, table: str, current: bool) -> Columns:
    """Build one bounded NumPy batch with exact integers and honest missing cells."""
    if not rows:
        return {}
    result: Columns = {}
    for name in rows[0]:
        values = [row.get(name) for row in rows]
        text = name in _TEXT or name.endswith("_status")
        integer = name in _INTS or (
            name in _EXACT and table in {"episodes", "matches"} and current
        )
        if (
            text
            or name in _NULLABLE
            or (integer and any(value is None for value in values))
            or (
                name in {"kd_ratio", "system_game_score"}
                and any(value is None for value in values)
            )
        ):
            result[name] = np.asarray(values, dtype=object)
        elif integer:
            result[name] = np.asarray(values, dtype=np.int64)
        else:
            dtype = (
                np.float32
                if current and table in {"priority_metrics", "full_metrics", "matches"}
                else np.float64
            )
            result[name] = np.asarray(
                [np.nan if value is None else value for value in values], dtype=dtype
            )
    return result


def _row_arrays(columns: Columns, order: Sequence[int] | None = None) -> Iterator[Row]:
    """Yield host scalar rows without duplicating an existing wide column table."""
    if not columns:
        return
    size = len(next(iter(columns.values())))
    if any(len(value) != size for value in columns.values()):
        raise ValueError("in-memory table columns have different lengths")
    for index in range(size) if order is None else order:
        yield {
            name: value[index].item()
            if isinstance(value[index], np.generic)
            else value[index]
            for name, value in columns.items()
        }


def _pass_facts(entry: Row, pass_complete: bool) -> Row:
    """Read one pass's random root and sampling facts for its scheduled games.

    Returns pass_complete (as given), root_seed, root_known (a uint32 root with
    a supported protocol) and sampling: stochastic, deterministic or unknown,
    combined from both teams' recorded method facts.
    """
    from marl_battlegrounds.evaluation.sampling_evidence import (
        combine_sampling_facts,
    )

    details = _mapping(entry.get("details"))
    root_seed = details.get("seed")
    facts = _mapping(details.get("method_sampling"))
    sampling = combine_sampling_facts(
        [facts[team] for team in ("team_a", "team_b")]
        if all(isinstance(facts.get(team), Mapping) for team in ("team_a", "team_b"))
        else []
    )
    return {
        "pass_complete": pass_complete,
        "root_seed": root_seed,
        "root_known": type(root_seed) is int
        and 0 <= root_seed < 2**32
        and details.get("rng_protocol")
        in ("episode-fold-in-v1", "evaluation-systems-v1"),
        "sampling": sampling["determinism"],
    }


def _interval_rule(reasons: Sequence[str]) -> tuple[bool, str, str | None, str | None]:
    """Decide whether a summary row shows its ranges, and how it labels them.

    One rule for ``point_margin_summary`` and ``system_metrics``. Elo rankings
    give unknown sampling the same label, but their
    ``conditional_interval_assumption`` describes their separate
    ``conditional_*`` columns; here it describes the row's own bounds. reasons
    are the row's de-duplicated problems, in order.

    Parameters
    ----------
    reasons : sequence of str
        Every reason the row's ranges could be wrong, such as "Sampling
        mechanism unknown" or "Requested games are incomplete". Empty means
        nothing is wrong.

    Returns
    -------
    tuple of (bool, str, str or None, str or None)
        (compute, interval_status, unavailable_reason, assumption):

        - No reasons: (True, "Available", None, None).
        - Only "Sampling mechanism unknown" (for example, a researcher's own
          System): the ranges are still calculated and shown, labeled
          "Conditional: Sampling not verified", with the assumption "Assumes
          declared game groups are independent; sampling is not verified".
          The reason is None, because nothing is withheld.
        - Anything else, alone or beside unknown sampling: (False,
          "Unavailable", every reason joined with "; ", None). The ranges are
          withheld; a conditional label never makes them valid.
    """
    if not reasons:
        return True, "Available", None, None
    if list(reasons) == ["Sampling mechanism unknown"]:
        return (
            True,
            CONDITIONAL_INTERVAL_STATUS,
            None,
            CONDITIONAL_INTERVAL_ASSUMPTION,
        )
    return False, "Unavailable", "; ".join(reasons), None


def _plan_record(
    origin: tuple[str, str, str, str],
    facts: Row,
    declared: Mapping[str, Any],
    row: Row | None,
    context: tuple[str, str, str, str, object],
) -> Row:
    """Describe one scheduled game for summaries that resample its pair.

    origin is the game's full ownership (run ID, phase, pass ID, episode ID as
    text); facts come from _pass_facts; declared is its schedule declaration;
    row is its completed outcome row or None; context is (Team A ID, Team B ID,
    their names, spawn choice). The record keeps those values plus map_id,
    config_id, cell (map and configuration as one key), map_key, pair (the
    full-origin verified spawn-pair key, or None), unit (that pair, or its
    declared bootstrap block) and seed (root seed and seed ID).
    """
    a, b, name_a, name_b, spawn = context
    map_id = declared.get("map_id")
    config_id = (
        declared.get("source_config_id")
        or declared.get("configuration_digest")
        or declared.get("config_id")
    )
    pair = declared.get("paired_comparison_key", declared.get("comparison_key"))
    pair_key = (
        (*origin[:3], str(pair))
        if declared.get("comparison_kind") == "verified_spawn_pair" and pair is not None
        else None
    )
    block = declared.get("bootstrap_group")
    return {
        "origin": origin,
        "row": row,
        "a": a,
        "b": b,
        "name_a": name_a,
        "name_b": name_b,
        "spawn": spawn,
        "map_id": map_id,
        "config_id": config_id,
        "cell": json.dumps([map_id, config_id], separators=(",", ":")),
        "map_key": f"map:{map_id}" if map_id is not None else f"config:{config_id}",
        "pass_complete": facts["pass_complete"],
        "root_known": facts["root_known"],
        "sampling": facts["sampling"],
        "pair": pair_key,
        "unit": None
        if pair_key is None
        else ("pair", *pair_key)
        if block is None
        else ("block", str(block)),
        "seed": (facts["root_seed"], declared.get("seed_id")),
    }


def _bootstrap_indices(
    sizes: Sequence[int], draws: int, seed: int
) -> list[list[NDArray[np.int64]]]:
    """Draw resampled unit positions for each stratum, in chunks of 64 draws.

    sizes gives each stratum's unit count, in the caller's sorted stratum order.
    One private NumPy generator seeded with seed draws, stratum by stratum and
    chunk by chunk, an int64 (chunk, units) array of positions with
    replacement. point_margin_summary and system_metrics share this order, so
    the same seed gives them the same draws.
    """
    random = np.random.default_rng(seed)
    return [
        [
            random.integers(size, size=(min(64, draws - start), size))
            for start in range(0, draws, 64)
        ]
        for size in sizes
    ]


def _directions(
    a: str, b: str, name_a: str, name_b: str, spawn: object
) -> tuple[tuple[str, str, str, str, str, str | None], ...]:
    """Give each recorded System's own view of one game.

    Returns (System ID, opponent ID, System name, opponent name, played_as,
    spawn_end) for recorded Team A first, then Team B; a System that played
    itself appears once, as Team A. played_as is "team_a" or "team_b".
    spawn_end names the map's original spawn bank the System started from
    ("team_a" or "team_b"), or None when the game's spawn choice was not saved.
    """
    known = type(spawn) is int and spawn in (0, 1)
    views: list[tuple[str, str, str, str, str, str | None]] = []
    for reverse in (False, True) if a != b else (False,):
        end = ("team_b" if bool(spawn) != reverse else "team_a") if known else None
        views.append(
            (b, a, name_b, name_a, "team_b", end)
            if reverse
            else (a, b, name_a, name_b, "team_a", end)
        )
    return tuple(views)


def _member_labels(registration: Mapping[str, Any]) -> list[str] | None:
    """Give each policy ID of a System with members a readable member label.

    registration is the System's saved record, with its ``components`` list.
    A component's label is its name when no other component of the System
    shares it, else "name #<policy_id>" (two checkpoints of one run share a
    name). Returns None when the System has fewer than two components
    (``scalar_reports.has_members``).
    """
    components = registration.get("components")
    if not isinstance(components, list):
        return None
    items = cast(list[object], components)
    if not has_members(len(items)):
        return None
    names = [str(_mapping(item).get("name", "")) for item in items]
    counts = Counter(names)
    return [
        name if counts[name] == 1 else f"{name} #{index}"
        for index, name in enumerate(names)
    ]


def _side_member(labels: Sequence[str | None], active: Sequence[object]) -> str | None:
    """Join one team's distinct member labels in slot order, or give None.

    labels and active hold the team's five slots from its own side. None when a
    slot is active without a label (its member was not recorded for the whole
    game, or its System has no members; ``_game_members`` labels only Systems
    with members) or no slot is active.
    """
    present = [label for label, alive in zip(labels, active, strict=True) if alive]
    if not present or any(label is None for label in present):
        return None
    return " / ".join(dict.fromkeys(cast(list[str], present)))


# Fact columns of a system_metrics summary row; ratios= names may not reuse them.
_SYSTEM_FACT_COLUMNS = frozenset(
    {
        "system_id",
        "system_name",
        "opponent_id",
        "opponent_name",
        "map_id",
        "config_id",
        "played_as",
        "spawn_end",
        "games",
        "scheduled_games",
        "completed_pairs",
        "elo",
        "elo_ci_low",
        "elo_ci_high",
        "rank",
        "confidence",
        "independent_units",
        "interval_status",
        "unavailable_reason",
        "conditional_interval_assumption",
        "sampling_unit",
        "bootstrap_draws",
        "bootstrap_seed",
        *(f"agent_{slot}_class_id" for slot in range(10)),
        *_MEMBER_COLUMNS,
    }
)
# Identity columns that exchange their team when a game is read from Team B.
_IDENTITY_TWINS = {
    "team_a_policy": "team_b_policy",
    "team_b_policy": "team_a_policy",
    **{
        f"agent_{slot}_{suffix}": f"agent_{(slot + 5) % 10}_{suffix}"
        for slot in range(10)
        for suffix in ("class_id", "active")
    },
}
# Columns per batch when combining games; bounds the float64 working arrays.
_COLUMN_BATCH = 1024


def _metric_slots(name: str) -> tuple[int, ...]:
    """Return the global agent slots a metric column describes, if any.

    name must be a catalog column. Agent, source-to-recipient and ally-pair
    columns give their subject slots (0-9); a team-to-recipient column gives its
    recipient slots; team-level and game-level columns give (). Callers use the
    slots to blank a column when a slot's class differs between games.
    """
    column = METRIC_COLUMNS_BY_NAME[name]
    if column.scope in ("agent", "source_recipient", "ally_pair"):
        return column.subjects
    if column.scope == "team_recipient":
        return column.subjects[1:]
    return ()


def _unit_counts(
    chunks: Sequence[NDArray[np.int64]], units: int
) -> NDArray[np.float64]:
    """Turn one stratum's drawn positions into (draws, units) draw counts.

    chunks are _bootstrap_indices' int64 (chunk, units) position arrays for
    this stratum, in draw order; units is the stratum's unit count. Returns a
    float64 (total draws, units) array whose cell says how often that draw
    picked that unit, so a matrix product with per-unit sums gives every draw's
    totals at once.
    """
    rows: list[NDArray[np.float64]] = []
    for chosen in chunks:
        size = len(chosen)
        flat = np.arange(size)[:, None] * units + chosen
        rows.append(
            np.bincount(flat.ravel(), minlength=size * units)
            .reshape(size, units)
            .astype(np.float64)
        )
    return np.concatenate(rows)


class _View:
    """Keep one manifest scope and bounded row sources for a result object.

    Saved views own no open streams until iteration. Memory views borrow existing
    column arrays and keep only narrow episode/order metadata alongside them.
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        *,
        run_dir: Path | None,
        phase: str | None = None,
        pass_id: str | None = None,
        memory: Mapping[str, object] | None = None,
    ) -> None:
        """Resolve one immutable manifest snapshot and describe available tables.

        The raw table headers follow the manifest's own scalar schema version.
        """
        _schema(manifest)
        self._headers = _headers(manifest)
        self.manifest = manifest
        self.run_dir = run_dir
        self._file_inodes: dict[str, int] = {}
        self._manifest_stamp = (
            None if run_dir is None else _stamp(run_dir / "run_details.json")
        )
        self.memory = {} if memory is None else dict(memory)
        self.selected = _select_passes(manifest, phase, pass_id)
        self.identities = {
            (entry["phase"], entry["pass_id"]) for entry in self.selected.values()
        }
        self._entries_by_identity = {
            (entry["phase"], entry["pass_id"]): entry
            for entry in self.selected.values()
        }
        tournament_keys = {
            key
            for key, entry in manifest["passes"].items()
            if entry.get("phase") == "tournament"
        }
        self.whole_tournament = bool(tournament_keys) and (
            tournament_keys.issubset(self.selected)
            or (
                manifest.get("tournament_reuse") is not None
                and any(
                    entry.get("pass_role") == "tournament_coordinator"
                    for entry in self.selected.values()
                )
            )
        )
        self.scope = _scope(self.selected, self.whole_tournament)
        states = {key: _state(entry, manifest) for key, entry in self.selected.items()}
        self.status = (
            "failed"
            if any(value[0] == "failed" for value in states.values())
            else "incomplete"
            if any(value[0] != "complete" for value in states.values())
            else "complete"
        )
        if self.whole_tournament and not manifest.get("tournament_summary"):
            self.status = "incomplete" if self.status != "failed" else "failed"
        inherited = _mapping(manifest.get("details"))
        if len(self.selected) == 1:
            inherited.update(
                _mapping(next(iter(self.selected.values())).get("details"))
            )
        configurations = _mapping(manifest.get("configurations"))
        for entry in self.selected.values():
            for identifier, content in _mapping(
                _mapping(entry.get("details")).get("configurations")
            ).items():
                if (
                    identifier in configurations
                    and configurations[identifier] != content
                ):
                    raise ValueError("conflicting recorded configuration content")
                configurations[identifier] = content
        self.metadata: Row = {
            **inherited,
            "run_id": manifest.get("run_id"),
            "scope": self.scope,
            "metric_schema_id": manifest["metric_schema_id"],
            "metric_schema_version": manifest["metric_schema_version"],
            "status": self.status,
            "pass_status": {
                key: {"status": value[0], "reason": value[1]}
                for key, value in states.items()
            },
            "systems": manifest.get("systems", {}),
            "configurations": configurations,
            "passes": self.selected,
        }
        if len(self.selected) == 1:
            evidence = next(iter(self.selected.values())).get("host_evidence")
            if evidence:
                self.metadata["host_evidence"] = evidence
        self.metadata["tables"] = {name: self._describe(name) for name in _TABLES}
        self.metadata["spawn_balance"] = self._spawn_balance()

    def _header(self, filename: str) -> tuple[str, ...]:
        """Read only a stored header, or use the manifest's schema for an empty table.

        A current (host schema 2) stored header must equal the header of the
        manifest's own scalar schema; otherwise ValueError is raised.
        """
        if self.run_dir is None:
            value = self.memory.get(filename.removesuffix(".csv"))
            if isinstance(value, dict):
                return tuple(cast(dict[str, Any], value))
            if isinstance(value, (tuple, list)) and value:
                return tuple(cast(Sequence[Row], value)[0])
            return self._headers.get(filename, ())
        boundary = _mapping(self.manifest.get("tables")).get(filename)
        if boundary is None or not boundary.get("durable_bytes"):
            return self._headers.get(filename, ())
        path = self.run_dir / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size < boundary["durable_bytes"]
        ):
            raise ValueError(
                f"committed result table is missing or truncated: {filename}"
            )
        self._file_inodes.setdefault(filename, path.stat().st_ino)
        with path.open("rb") as stream:
            header = stream.readline(boundary["durable_bytes"])
        if not header.endswith(b"\n"):
            raise ValueError(f"result table has an incomplete header: {filename}")
        try:
            names = tuple(next(csv.reader([header.decode("utf-8")], strict=True)))
        except (ValueError, UnicodeError, csv.Error) as error:
            raise ValueError(f"invalid result table header: {filename}") from error
        if (
            not names
            or len(set(names)) != len(names)
            or any(not name for name in names)
        ):
            raise ValueError(f"invalid result table header: {filename}")
        if (
            self.manifest["schema_version"] == 2
            and filename in self._headers
            and names != self._headers[filename]
        ):
            raise ValueError(f"incompatible current result table header: {filename}")
        return names

    def _present(self, filename: str) -> bool:
        """Test recorded table presence without treating a zero value as missing."""
        if self.run_dir is None:
            return filename.removesuffix(".csv") in self.memory
        return filename in self.manifest.get("tables", {})

    def _describe(self, name: str) -> Row:
        """Declare table columns, availability and scope without reading wide rows."""
        filename = (
            "tournament_results.csv" if name == "tournament_rankings" else _FILES[name]
        )
        summary = name in _SUMMARIES
        availability, reason = "available", None
        current = self.manifest["schema_version"] == 2
        selected_modes = [
            _mapping(entry.get("details")).get("metrics")
            for entry in self.selected.values()
            if entry.get("pass_role") != "tournament_coordinator"
        ]
        measured = any(
            value in {"priority", "full"}
            for entry in self.selected.values()
            for value in _mapping(entry.get("recorded_metrics_by_episode")).values()
        )
        has_full = any(
            value == "full"
            for entry in self.selected.values()
            for value in _mapping(entry.get("recorded_metrics_by_episode")).values()
        )
        requested_full = any(
            _mapping(entry.get("details")).get("full_metrics_episodes")
            for entry in self.selected.values()
        )
        present = self._present(filename)
        if name in {"episodes", "priority_metrics"}:
            present |= self._present("match_results.csv") and any(
                entry["phase"] == "tournament" for entry in self.selected.values()
            )
        members = self._member_systems() if name == "policy_assignments" else {}
        if name == "policy_assignments":
            if self.run_dir is None:
                availability, reason = (
                    ("unavailable", _IN_MEMORY_MEMBERS)
                    if members
                    else ("disabled", _NO_MEMBERS)
                )
            elif not present and not members:
                availability, reason = "disabled", _NO_MEMBERS
        elif summary and not self.whole_tournament:
            availability, reason = (
                "unavailable",
                "This scope does not contain the complete tournament population",
            )
        elif summary and self.status != "complete":
            availability, reason = (
                "unavailable",
                "The complete tournament has not been finalized",
            )
        elif (
            name == "tournament_headline_metrics"
            and selected_modes
            and all(mode == "none" for mode in selected_modes)
        ):
            availability, reason = "disabled", "Tournament metrics were disabled"
        elif (
            name == "priority_metrics"
            and selected_modes
            and all(mode == "none" for mode in selected_modes)
            and not measured
            and (
                current
                or (not requested_full and not self._present("priority_metrics.csv"))
            )
        ):
            availability, reason = "disabled", "Priority metrics were disabled"
        elif (
            name == "full_metrics"
            and not has_full
            and not requested_full
            and selected_modes
            and all(mode != "full" for mode in selected_modes)
            and (current or not present)
        ):
            availability, reason = "disabled", "No full metric episodes were selected"
        elif not present:
            if (
                current
                and not summary
                and name in {"episodes", "priority_metrics", "full_metrics"}
                and not any(
                    entry.get("completed_episode_ids")
                    for entry in self.selected.values()
                )
            ):
                availability = "available"
            else:
                availability, reason = (
                    "unavailable",
                    "This table was not recorded in the selected experiment",
                )
        if name == "matches" and not any(
            entry["phase"] == "tournament" for entry in self.selected.values()
        ):
            availability, reason = (
                "unavailable",
                "This scope does not contain tournament games",
            )
        columns = self._header(filename)
        if name == "policy_assignments":
            columns = (*columns, "system_name", "member")
        if name == "episodes":
            columns = (*EPISODE_COLUMNS, "system_game_score")
        elif (
            name == "priority_metrics"
            and not self._present(filename)
            and self._present("match_results.csv")
        ):
            columns = (
                *IDENTITY_COLUMNS,
                *(
                    value
                    for value in self._header("match_results.csv")
                    if value in PRIORITY_METRIC_NAMES
                    or (
                        not current
                        and value in {f"agent_{slot}_return" for slot in range(10)}
                    )
                ),
            )
        if summary:
            if name == "tournament_rankings":
                columns = (
                    *columns,
                    *(
                        value
                        for value in ("system_id", "system_name", "rank")
                        if value not in columns
                    ),
                )
            columns = (
                *columns,
                *(
                    value
                    for value in ("run_id", "phase", "pass_id")
                    if value not in columns
                ),
            )
        missing_fields = [
            column
            for column in EPISODE_COLUMNS
            if name == "episodes" and column not in self._header(filename)
        ]
        if name == "episodes" and any(
            _mapping(_mapping(entry.get("details")).get("evaluation_contract")).get(
                "version"
            )
            != 1
            and _mapping(entry.get("details")).get("first_system_team")
            not in {"team_a", "team_b"}
            for entry in self.selected.values()
        ):
            missing_fields.append("system_game_score")
        if name == "tournament_rankings" and not self.manifest.get("systems"):
            missing_fields.append("system_id")
        extra: Row = (
            {"member_coverage": self._member_coverage(members)} if members else {}
        )
        return {
            **extra,
            "availability": availability,
            "reason": reason,
            "columns": columns,
            "column_types": {
                column: _cell_type(column, name, current) for column in columns
            },
            "scope": self.scope,
            "missing_historical_fields": tuple(missing_fields),
            "unavailable_columns": {
                column: "The original recorded identity or value is unavailable"
                for column in missing_fields
            },
            "missing_passes": tuple(
                (entry["phase"], entry["pass_id"])
                for entry in self.selected.values()
                if self.manifest["schema_version"] == 1
                and name in {"priority_metrics", "full_metrics"}
                and not entry.get("recorded_metrics_by_episode")
                and _mapping(entry.get("details")).get("metrics") is None
            ),
        }

    def _registration(self, row: Mapping[str, Any]) -> Row:
        """Return the saved System record that owns one assignment row."""
        systems = _mapping(self.manifest.get("systems"))
        return _mapping(systems.get(str(row["system_id"])))

    def _member_systems(self) -> dict[str, Row]:
        """Map each System with members in this scope to its saved record.

        Reads only the manifest. Returns {} at once, without visiting games,
        when no registered System has members.
        """
        systems = _mapping(self.manifest.get("systems"))
        labels = {
            str(system): _mapping(registration)
            for system, registration in systems.items()
            if _member_labels(_mapping(registration)) is not None
        }
        if not labels:
            return {}
        used: set[str] = set()
        for entry in self.selected.values():
            used.update(map(str, _mapping(entry.get("system_ids")).values()))
            for declaration in _mapping(entry.get("episodes")).values():
                owners = _mapping(_mapping(declaration).get("system_ids"))
                used.update(map(str, owners.values()))
        return {system: value for system, value in labels.items() if system in used}

    def _member_coverage(self, members: Mapping[str, Row]) -> Row:
        """Count each member System's completed games with and without members.

        A completed game is recorded when its pass lists it in
        ``trace_epochs``, as summary and per-decision rows both do. Returns
        {System ID: {"system_name": name, "recorded": n, "not_recorded": m}}
        from the manifest only.
        """
        counts: dict[str, dict[str, Any]] = {
            system: {
                "system_name": registration.get("name"),
                "recorded": 0,
                "not_recorded": 0,
            }
            for system, registration in members.items()
        }
        for entry in self.selected.values():
            epochs = _mapping(entry.get("trace_epochs"))
            default = _mapping(entry.get("system_ids"))
            declarations = _mapping(entry.get("episodes"))
            for episode in entry.get("completed_episode_ids", ()):
                owners = (
                    _mapping(_mapping(declarations.get(str(episode))).get("system_ids"))
                    or default
                )
                recorded = "recorded" if str(episode) in epochs else "not_recorded"
                for system in set(map(str, owners.values())) & counts.keys():
                    counts[system][recorded] += 1
        return counts

    def _game_members(self) -> dict[tuple[str, str, str, str], tuple[str | None, ...]]:
        """Label each recorded game's ten slots with the member that played it.

        Keys are (run ID, phase, pass ID, episode ID) as text. A slot has a label
        only when its System has members and one row with a policy ID of 0 or
        more covers the whole game, decisions 0 to its trace epoch; otherwise
        None. Reads the assignment table once; {} when nothing applies.
        """
        members = self._member_systems()
        description = self.metadata["tables"]["policy_assignments"]
        if not members or description["availability"] != "available":
            return {}
        rows: dict[tuple[str, str, str, str], list[Row | None]] = {}
        seen: set[tuple[tuple[str, str, str, str], int]] = set()
        for row in self.iter_rows("policy_assignments", 1024):
            system = str(row["system_id"])
            key = (
                str(row["run_id"]),
                str(row["phase"]),
                str(row["pass_id"]),
                str(row["episode_id"]),
            )
            slot = int(row["global_slot"])
            slots = rows.setdefault(key, [None] * 10)
            if (key, slot) in seen:
                # A second interval: the member did not cover the whole game.
                slots[slot] = None
                continue
            seen.add((key, slot))
            epoch = _mapping(self._entry(row).get("trace_epochs")).get(key[3])
            if (
                system in members
                and int(row["policy_id"]) >= 0
                and int(row["decision_start"]) == 0
                and isinstance(epoch, int)
                and int(row["decision_stop"]) == epoch + 1
            ):
                slots[slot] = row
        return {
            key: tuple(None if row is None else row["member"] for row in slots)
            for key, slots in rows.items()
        }

    def _check_snapshot(self, filename: str) -> None:
        """Reject destructive recovery while allowing append-only later commits."""
        if self.run_dir is None:
            return
        before = _mapping(self.manifest.get("tables")).get(
            filename, {"durable_bytes": 0, "rows": 0}
        )
        if before["durable_bytes"]:
            path = self.run_dir / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("result snapshot table is missing or unsafe")
            table_stat = path.stat()
            if table_stat.st_size < before[
                "durable_bytes"
            ] or table_stat.st_ino != self._file_inodes.setdefault(
                filename, table_stat.st_ino
            ):
                raise ValueError("result snapshot table was truncated or replaced")
        stamp = _stamp(self.run_dir / "run_details.json")
        if stamp == self._manifest_stamp:
            return
        current = _read_manifest(self.run_dir)
        if current.get("run_id") != self.manifest.get("run_id"):
            raise ValueError("saved result identity changed during reading")
        after = _mapping(current.get("tables")).get(
            filename, {"durable_bytes": 0, "rows": 0}
        )
        if after.get("durable_bytes", 0) < before.get("durable_bytes", 0) or after.get(
            "rows", 0
        ) < before.get("rows", 0):
            raise ValueError("result snapshot was truncated or restored while reading")
        for key, entry in self.selected.items():
            now = _mapping(current.get("passes")).get(key)
            if now is None or not set(entry.get("completed_episode_ids", ())).issubset(
                now.get("completed_episode_ids", ())
            ):
                raise ValueError("selected result pass was restored while reading")
            for identifier, declaration in _mapping(entry.get("episodes")).items():
                if _mapping(now.get("episodes")).get(identifier) != declaration:
                    raise ValueError("selected result conditions changed while reading")
        if (
            self.manifest.get("tournament_summary") != current.get("tournament_summary")
            and self.manifest.get("tournament_summary") is not None
        ):
            raise ValueError("selected tournament summary changed while reading")
        self._manifest_stamp = stamp

    def _raw(
        self, filename: str, rows: int, columns: frozenset[str] | None = None
    ) -> Iterator[Row]:
        """Stream scoped rows from an existing memory table or durable CSV prefix.

        columns keeps only those cells of a raw game table and skips converting
        the rest; None keeps every cell. Summary tables ignore it.
        """
        if not self._present(filename):
            return
        if self.run_dir is None:
            data = self.memory[filename.removesuffix(".csv")]
            if isinstance(data, dict):
                order = None
                arrays = cast(Columns, data)
                if "episode_id" in arrays and "completion_order" in self.memory:
                    indexes = {
                        int(value): index
                        for index, value in enumerate(arrays["episode_id"])
                    }
                    order = tuple(
                        indexes[value]
                        for value in cast(
                            Sequence[int], self.memory["completion_order"]
                        )
                        if value in indexes
                    )
                iterator = _row_arrays(arrays, order)
            else:
                sequence = cast(Sequence[Row], data)
                if filename in self._headers and "completion_order" in self.memory:
                    indexed = {int(row["episode_id"]): row for row in sequence}
                    iterator = (
                        indexed[value]
                        for value in cast(
                            Sequence[int], self.memory["completion_order"]
                        )
                        if value in indexed
                    )
                else:
                    iterator = iter(sequence)
            for row in iterator:
                if (
                    filename not in self._headers
                    or (row.get("phase"), row.get("pass_id")) in self.identities
                ):
                    yield (
                        dict(row)
                        if columns is None or filename not in self._headers
                        else {key: row[key] for key in row if key in columns}
                    )
            return
        summary = filename not in self._headers
        if filename == "policy_assignments.csv":
            for batch in iter_policy_assignments(
                self.run_dir / filename,
                manifest=self.manifest,
                expected_header=ASSIGNMENT_COLUMNS,
                batch_size=rows,
            ):
                self._check_snapshot(filename)
                for row in batch:
                    if (row.get("phase"), row.get("pass_id")) in self.identities:
                        yield dict(row)
            return
        batches = (
            iter_summary_rows(
                self.run_dir / filename,
                manifest=self.manifest,
                expected_header=self._headers.get(filename),
                batch_size=rows,
            )
            if summary
            else iter_scalar_rows(
                self.run_dir / filename,
                manifest=self.manifest,
                expected_header=self._headers.get(filename),
                batch_size=rows,
                columns=columns,
            )
        )
        for batch in batches:
            self._check_snapshot(filename)
            for row in batch:
                if summary or (row.get("phase"), row.get("pass_id")) in self.identities:
                    yield dict(row)

    def _entry(self, row: Mapping[str, Any]) -> Row:
        """Find the exact phase/pass owner of one recorded row."""
        entry = self._entries_by_identity.get((row.get("phase"), row.get("pass_id")))
        if entry is None:
            raise ValueError("result row has no selected pass owner")
        return entry

    def _episodes(self, rows: int) -> Iterator[Row]:
        """Project exact required outcomes while keeping each raw authority's order."""
        for filename in ("episodes.csv", "match_results.csv"):
            for row in self._raw(filename, rows):
                result = {name: row.get(name) for name in EPISODE_COLUMNS}
                entry = self._entry(row)
                details = _mapping(entry.get("details"))
                new_contract = (
                    _mapping(details.get("evaluation_contract")).get("version") == 1
                )
                binding = details.get("first_system_team")
                # Only new fixed-team contracts establish Team A as the focal
                # submitted System. Old records must carry their actual binding.
                team = "team_a" if new_contract else binding
                outcome = row.get("outcome")
                result["system_game_score"] = (
                    None
                    if team not in {"team_a", "team_b"} or outcome not in {1, 2, 3}
                    else 0.5
                    if outcome == 3
                    else float(outcome == (1 if team == "team_a" else 2))
                )
                yield result

    def _match_context(
        self, row: Mapping[str, Any]
    ) -> tuple[str, str, str, str, object]:
        """Return recorded Team A/B identities, display names and spawn choice.

        row keeps its complete original ownership. Missing historical System
        identities raise ValueError; display names never replace identity.
        Spawn choice is 0, 1 or an unavailable value from the episode declaration.
        """
        entry = self._entry(row)
        ids = _mapping(entry.get("system_ids"))
        a, b = ids.get("team_a"), ids.get("team_b")
        if not isinstance(a, str) or not a or not isinstance(b, str) or not b:
            raise ValueError(
                "Unavailable: these saved games lack their original System IDs"
            )
        declaration = _mapping(
            _mapping(entry.get("episodes")).get(str(row["episode_id"]))
        )
        return (
            a,
            b,
            str(row.get("team_a_policy") or a),
            str(row.get("team_b_policy") or b),
            declaration.get("spawn_locations"),
        )

    def participant_names(self) -> dict[str, str]:
        """List exact selected System IDs, including declared games not yet played."""
        names: dict[str, str] = {}
        for entry in self.selected.values():
            ids = _mapping(entry.get("system_ids"))
            policies = _mapping(entry.get("policies"))
            for team, identifier in ids.items():
                if isinstance(identifier, str) and identifier:
                    names[identifier] = str(
                        _mapping(policies.get(team)).get("name") or identifier
                    )
        return names

    def _game_plan(self) -> list[Row]:
        """List every scheduled game in scope with its sides and sampling facts.

        One dict per declared game, in schedule order (see _plan_record for the
        fields). Completed outcomes come from one narrow read of the episodes
        table. The canonical view overrides this with its logical schedule, so
        reused games keep their original pass declarations.
        """
        completed = {
            (row["phase"], row["pass_id"], str(row["episode_id"])): row
            for row in self.iter_rows("episodes", 128)
        }
        run_id = str(self.manifest.get("run_id"))
        plan: list[Row] = []
        for entry in self.selected.values():
            schedule = _mapping(entry.get("episodes"))
            if not schedule:
                continue
            facts = _pass_facts(entry, _state(entry, self.manifest)[0] == "complete")
            for identifier, raw in schedule.items():
                row = completed.get((entry["phase"], entry["pass_id"], str(identifier)))
                probe = row or {
                    "phase": entry["phase"],
                    "pass_id": entry["pass_id"],
                    "episode_id": identifier,
                }
                plan.append(
                    _plan_record(
                        (run_id, entry["phase"], entry["pass_id"], str(identifier)),
                        facts,
                        _mapping(raw),
                        row,
                        self._match_context(probe),
                    )
                )
        return plan

    def point_margin_summary(self, by: str, draws: int, seed: int) -> list[Row]:
        """Summarize Team A points using complete declared spawn-pair evidence.

        by, draws and seed are checked by the public accessor. Read narrow saved
        outcomes once and retain their narrow rows and per-game declarations.
        Exact identities group rows; the declared schedule keeps missing games
        visible. Pair keys carry each game's full origin (run, phase, pass and
        pair), so reused canonical games never merge with local ones; explicit
        larger block keys keep their declared scope. Incomplete evidence keeps
        descriptive means but withholds intervals; unknown sampling alone keeps
        them, labeled conditional (``_interval_rule``). Sampling is judged once
        per row, as in ``system_metrics``: unknown wins, and "No random
        sampling variation" applies only when no game is stochastic. No model
        or file is changed.
        """
        groups: dict[tuple[str, str, str | None], Row] = {}
        for game in self._game_plan():
            row = game["row"]
            map_id, config_id = game["map_id"], game["config_id"]
            key = (game["a"], game["b"], game["map_key"] if by == "map" else None)
            group = groups.setdefault(
                key,
                {
                    "system_id": game["a"],
                    "opponent_id": game["b"],
                    "system_name": game["name_a"],
                    "opponent_name": game["name_b"],
                    "map_id": map_id if by == "map" else None,
                    "config_id": config_id if by == "map" and map_id is None else None,
                    "scheduled_games": 0,
                    "games": 0,
                    "values": [],
                    "pairs": {},
                    "reasons": [],
                    "samplings": set(),
                },
            )
            group["scheduled_games"] += 1
            if not game["pass_complete"]:
                group["reasons"].append("Requested pass is incomplete or failed")
            if not game["root_known"]:
                group["reasons"].append("Random root coordinates are unavailable")
            group["samplings"].add(game["sampling"])
            margin = None
            if row is None:
                group["reasons"].append("Requested games are incomplete")
            else:
                group["games"] += 1
                points = (row.get("team_a_score"), row.get("team_b_score"))
                if all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in points
                ):
                    margin = float(cast(float, points[0])) - float(
                        cast(float, points[1])
                    )
                    if not math.isfinite(margin):
                        margin = None
                else:
                    margin = None
                if margin is None:
                    group["reasons"].append("Native points are missing")
                else:
                    group["values"].append(margin)
            if game["pair"] is None:
                group["reasons"].append(
                    "Verified exchanged-spawn pairs are unavailable"
                )
                continue
            group["pairs"].setdefault(game["pair"], []).append(
                {
                    "cell": game["cell"],
                    "spawn": game["spawn"],
                    "unit": game["unit"],
                    # Both supported protocols use the same actor-key
                    # coordinates; their labels do not create independence.
                    "seed": game["seed"],
                    "margin": margin if row is not None else None,
                }
            )
        result: list[Row] = []
        for group in groups.values():
            units: dict[tuple[str, ...], list[Row]] = defaultdict(list)
            complete_pairs = 0
            seed_units: dict[tuple[object, ...], set[tuple[str, ...]]] = defaultdict(
                set
            )
            for pair in group["pairs"].values():
                if (
                    len(pair) != 2
                    or {row["spawn"] for row in pair} != {0, 1}
                    or len({row["cell"] for row in pair}) != 1
                    or len({row["unit"] for row in pair}) != 1
                    or len({row["seed"] for row in pair}) != 1
                    or pair[0]["seed"][-1] is None
                    or any(row["margin"] is None for row in pair)
                ):
                    group["reasons"].append(
                        "An exchanged-spawn pair is incomplete or inconsistent"
                    )
                    continue
                complete_pairs += 1
                units[pair[0]["unit"]].extend(pair)
                seed_units[pair[0]["seed"]].add(pair[0]["unit"])
            if any(len(value) > 1 for value in seed_units.values()):
                group["reasons"].append(
                    "Repeated seed coordinates need one declared sampling block"
                )
            # Resample whole blocks with the same covered cells together.
            # Counts may differ: use their sampled point sums/game counts, then
            # preserve the original cell weights when combining means.
            strata: dict[tuple[str, ...], list[dict[str, tuple[float, int]]]] = (
                defaultdict(list)
            )
            for rows in units.values():
                by_cell: dict[str, list[float]] = defaultdict(list)
                for row in rows:
                    by_cell[row["cell"]].append(row["margin"])
                summary = {
                    cell: (math.fsum(values), len(values))
                    for cell, values in by_cell.items()
                }
                strata[tuple(sorted(by_cell))].append(summary)
            if not strata or any(len(values) < 2 for values in strata.values()):
                group["reasons"].append(
                    "Fewer than two independent complete units in a condition stratum"
                )
            if complete_pairs * 2 != group["scheduled_games"]:
                group["reasons"].append("Requested pair evidence is incomplete")
            # One sampling verdict per row, as in system_metrics: unknown
            # wins, and a row without any stochastic game cannot vary.
            samplings = group.pop("samplings")
            first = (
                "Requested pass is incomplete or failed",
                "Random root coordinates are unavailable",
            )
            seen = list(dict.fromkeys(group["reasons"]))
            reasons = [reason for reason in seen if reason in first]
            if "unknown" in samplings:
                reasons.append("Sampling mechanism unknown")
            elif "stochastic" not in samplings:
                reasons.append("No random sampling variation")
            reasons += [reason for reason in seen if reason not in first]
            compute, status, reason, assumption = _interval_rule(reasons)
            low, high = None, None
            if compute:
                ordered = sorted(strata.items())
                indices = _bootstrap_indices(
                    [len(blocks) for _, blocks in ordered], draws, seed
                )
                sampled = np.zeros(draws, np.float64)
                for (cells, blocks), chunks in zip(ordered, indices, strict=True):
                    columns = [
                        (
                            np.asarray(
                                [block[cell][0] for block in blocks], np.float64
                            ),
                            np.asarray(
                                [block[cell][1] for block in blocks], np.float64
                            ),
                        )
                        for cell in cells
                    ]
                    for start, chosen in zip(range(0, draws, 64), chunks, strict=True):
                        size = len(chosen)
                        for sums, counts in columns:
                            weight = counts.sum() / group["games"]
                            means = sums[chosen].sum(axis=1) / counts[chosen].sum(
                                axis=1
                            )
                            sampled[start : start + size] += means * weight
                bounds = np.percentile(sampled, (2.5, 97.5), method="linear")
                low, high = float(bounds[0]), float(bounds[1])
            values = group.pop("values")
            group.pop("pairs")
            group.pop("reasons")
            result.append(
                {
                    **group,
                    "completed_pairs": complete_pairs,
                    "independent_units": (
                        0
                        if "No random sampling variation" in reasons
                        else None
                        if any(
                            reason in reasons
                            for reason in (
                                "Sampling mechanism unknown",
                                "Random root coordinates are unavailable",
                                "Repeated seed coordinates need "
                                "one declared sampling block",
                            )
                        )
                        else len(units)
                    ),
                    "mean_point_margin": math.fsum(
                        value / len(values) for value in values
                    )
                    if values and len(values) == group["games"]
                    else None,
                    "ci_low": low,
                    "ci_high": high,
                    "confidence": 0.95,
                    "interval_status": status,
                    "unavailable_reason": reason,
                    "conditional_interval_assumption": assumption,
                    "sampling_unit": "Whole declared blocks of exchanged-spawn pairs",
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed,
                }
            )
        return result

    def head_to_head(self, by: str) -> list[Row]:
        """Reduce narrow saved outcomes once into directed descriptive matchup rows.

        by is checked by the public accessor. Counts and point means weight each
        completed game equally. No metric report or statistical fitter is read.
        Full tournament matchup intervals are copied only for an exact overall
        name/count match with one exact identity pair; all other bounds remain
        unavailable.
        """
        groups: dict[tuple[str, str, object], Row] = {}
        for row in self.iter_rows("episodes", 128):
            a, b, name_a, name_b, spawn = self._match_context(row)
            if by == "spawn" and (type(spawn) is not int or spawn not in (0, 1)):
                raise ValueError(
                    "Spawn head-to-head unavailable: a game's spawn end was not saved"
                )
            map_id = row.get("map_id")
            condition = (map_id, row.get("config_id") if map_id is None else None)
            for view in _directions(a, b, name_a, name_b, spawn):
                system, opponent, system_name, opponent_name, played_as, side = view
                reverse = played_as == "team_b"
                names = (system_name, opponent_name)
                end = side if by == "spawn" else None
                key = (system, opponent, condition if by == "map" else end)
                if key not in groups:
                    groups[key] = {
                        "system_id": system,
                        "system_name": names[0],
                        "opponent_id": opponent,
                        "opponent_name": names[1],
                        **(
                            {"map_id": map_id, "config_id": condition[1]}
                            if by == "map"
                            else {}
                        ),
                        **({"spawn_end": end} if by == "spawn" else {}),
                        "games": 0,
                        "wins": 0,
                        "draws": 0,
                        "losses": 0,
                        "_outcomes": True,
                        "_points_for": 0,
                        "_points_against": 0,
                        "expected_score_ci_low": None,
                        "expected_score_ci_high": None,
                        "expected_score_interval_status": "unavailable",
                    }
                value = groups[key]
                value["games"] += 1
                outcome = row.get("outcome")
                if outcome in (1, 2, 3):
                    field = (
                        "draws"
                        if outcome == 3
                        else "wins"
                        if outcome == (2 if reverse else 1)
                        else "losses"
                    )
                    value[field] += 1
                else:
                    value["_outcomes"] = False
                for field, team in (
                    ("_points_for", "b" if reverse else "a"),
                    ("_points_against", "a" if reverse else "b"),
                ):
                    score = row.get(f"team_{team}_score")
                    if value[field] is not None:
                        value[field] = (
                            value[field] + score
                            if isinstance(score, (int, float))
                            and not isinstance(score, bool)
                            and np.isfinite(score)
                            else None
                        )
        result: list[Row] = []
        for key in sorted(
            groups, key=lambda value: (value[0], value[1], repr(value[2]))
        ):
            value = groups[key]
            games = value["games"]
            outcomes = value.pop("_outcomes")
            for field in ("wins", "draws", "losses"):
                if not outcomes:
                    value[field] = None
                value[
                    {"wins": "win_rate", "draws": "draw_rate", "losses": "loss_rate"}[
                        field
                    ]
                ] = value[field] / games if outcomes else np.nan
            value["expected_score"] = (
                (value["wins"] + 0.5 * value["draws"]) / games if outcomes else np.nan
            )
            points_for, points_against = (
                value.pop("_points_for"),
                value.pop("_points_against"),
            )
            value["mean_points_for"] = (
                points_for / games if points_for is not None else np.nan
            )
            value["mean_points_against"] = (
                points_against / games if points_against is not None else np.nan
            )
            value["mean_point_margin"] = (
                (points_for - points_against) / games
                if points_for is not None and points_against is not None
                else np.nan
            )
            result.append(value)
        if (
            by == "overall"
            and self.metadata["tables"]["matchup_results"]["availability"]
            == "available"
        ):
            by_names: dict[tuple[str, str], list[Row]] = {}
            for value in result:
                by_names.setdefault(
                    (value["system_name"], value["opponent_name"]), []
                ).append(value)
            for saved in self.iter_rows("matchup_results", 128):
                policy, opponent = saved.get("policy"), saved.get("opponent")
                if not isinstance(policy, str) or not isinstance(opponent, str):
                    continue
                matches = by_names.get((policy, opponent), [])
                if len(matches) == 1 and matches[0]["games"] == saved.get("matches"):
                    for name in (
                        "expected_score_ci_low",
                        "expected_score_ci_high",
                        "expected_score_interval_status",
                    ):
                        matches[0][name] = saved.get(name)
        return result

    def system_metrics(
        self,
        table: str,
        by: str,
        systems: Iterable[str] | None,
        columns: Iterable[str] | None,
        ratios: Mapping[str, tuple[str, str]],
        draws: int,
        seed: int,
    ) -> Columns:
        """Read a metric table once and show every System as Team A.

        The public accessor checks argument types and owns the contract. This
        method checks names against the table, reads only the needed columns,
        reads Team B games through mirror_metric, and either returns one row per
        System per game or combines games with each column's summary weight and
        resamples whole verified pairs for intervals. Host-only NumPy.
        """
        description = self.metadata["tables"][table]
        if description["availability"] == "disabled":
            return {}
        if description["availability"] != "available":
            raise ValueError(description["reason"])
        present = [
            name for name in description["columns"] if name in METRIC_COLUMNS_BY_NAME
        ]
        known = set(present)
        requested = list(present if columns is None else columns)
        absent = [name for name in requested if name not in known]
        if absent:
            raise ValueError(
                f"These are not metric columns of the {table} table: "
                + ", ".join(absent[:5])
            )
        for name, parts in ratios.items():
            if (
                name in METRIC_COLUMNS_BY_NAME
                or name in IDENTITY_COLUMNS
                or name in _SYSTEM_FACT_COLUMNS
                or name.endswith(("_ci_low", "_ci_high"))
            ):
                raise ValueError(f"The ratio name {name!r} is already a column name")
            missing = [part for part in parts if part not in known]
            if missing:
                raise ValueError(
                    f"Ratio {name!r} uses columns not in the {table} table: "
                    + ", ".join(missing)
                )
        weight_of = {
            name: weight
            if (weight := METRIC_COLUMNS_BY_NAME[name].summary_weight) in known
            else None
            for name in requested
        }
        sources = [
            *requested,
            *(part for parts in ratios.values() for part in parts),
            *(weight for weight in weight_of.values() if weight is not None),
        ]
        twin = {
            name: mirror_metric(name)
            for source in sources
            for name in (source, mirror_metric(source)[0])
        }
        order = sorted(twin)
        index = {name: position for position, name in enumerate(order)}

        names = self.participant_names()
        wanted: set[str] | None = None
        if systems is not None:
            wanted = set()
            for item in systems:
                if item in names:
                    wanted.add(item)
                    continue
                matches = sorted(key for key, value in names.items() if value == item)
                if not matches:
                    raise ValueError(f"No System here has the ID or name {item!r}")
                if len(matches) > 1:
                    raise ValueError(
                        f"{item!r} names several Systems; use one of their IDs: "
                        + ", ".join(matches)
                    )
                wanted.add(matches[0])

        # Older runs without an episodes table still read, without pairs.
        plan = (
            self._game_plan()
            if self.metadata["tables"]["episodes"]["availability"] == "available"
            else []
        )
        by_origin = {record["origin"]: record for record in plan}
        values = np.full((max(len(plan), 1), len(order)), np.nan, np.float32)
        games: list[tuple[Row, Row]] = []
        unknown_facts: Row = {
            "pass_complete": True,
            "root_seed": None,
            "root_known": False,
            "sampling": "unknown",
        }
        projection = frozenset((*IDENTITY_COLUMNS, *order))
        cells = itemgetter(*order) if len(order) > 1 else None
        pending: list[tuple[object, ...]] = []

        def store() -> None:
            """Convert the waiting rows to float32 in one step; blank means NaN."""
            nonlocal values
            if not pending:
                return
            block = np.array(pending, dtype=object).reshape(len(pending), len(order))
            block[np.equal(block, cast(Any, None))] = np.nan
            first = len(games) - len(pending)
            while len(values) < len(games):
                values = np.concatenate([values, np.full_like(values, np.nan)])
            values[first : len(games)] = block.astype(np.float32)
            pending.clear()

        for row in self.iter_rows(table, 256, projection):
            origin = tuple(
                str(row.get(name))
                for name in ("run_id", "phase", "pass_id", "episode_id")
            )
            record = by_origin.get(origin)
            if record is None:
                # A game the schedule does not declare (older runs): its sides
                # still come from the run, but no pair or sampling is known.
                record = _plan_record(
                    cast(tuple[str, str, str, str], origin),
                    unknown_facts,
                    {"map_id": row.get("map_id"), "config_id": row.get("config_id")},
                    row,
                    self._match_context(row),
                )
                plan.append(record)
                by_origin[record["origin"]] = record
            pending.append(
                cast(tuple[object, ...], cells(row))
                if cells is not None
                else (row.get(order[0]),)
            )
            games.append((record, {name: row.get(name) for name in IDENTITY_COLUMNS}))
            if len(pending) == 256:
                store()
        store()
        values = values[: len(games)]

        def oriented(
            sources: Sequence[str | None],
            positions: NDArray[np.int64],
            mirrored: NDArray[np.bool_],
            dtype: type[np.floating[Any]] = np.float64,
        ) -> NDArray[np.floating[Any]]:
            """Read sources for these games from each game's System side.

            A None source reads as 1. Team B games read each source's twin,
            with its sign. Rows and columns are taken in one step, so only the
            requested cells are copied.
            """
            result = np.ones((len(positions), len(sources)), dtype)
            real = np.asarray(
                [place for place, name in enumerate(sources) if name is not None],
                np.int64,
            )
            if not len(real) or not len(positions):
                return result
            own = np.asarray(
                [index[cast(str, sources[place])] for place in real], np.int64
            )
            other = np.asarray(
                [index[twin[cast(str, sources[place])][0]] for place in real], np.int64
            )
            signs = np.asarray(
                [twin[cast(str, sources[place])][1] for place in real], dtype
            )
            straight = np.flatnonzero(~mirrored)
            flipped = np.flatnonzero(mirrored)
            result[np.ix_(straight, real)] = values[np.ix_(positions[straight], own)]
            # Adding zero turns a mirrored -0.0 into 0.0.
            result[np.ix_(flipped, real)] = (
                values[np.ix_(positions[flipped], other)] * signs + 0.0
            )
            return result

        if by == "game":
            fact_rows: list[Row] = []
            positions_list: list[int] = []
            mirrored_list: list[bool] = []
            game_members = self._game_members()
            unlabeled: tuple[str | None, ...] = (None,) * 10
            for position, (record, identity) in enumerate(games):
                for view in _directions(
                    record["a"],
                    record["b"],
                    record["name_a"],
                    record["name_b"],
                    record["spawn"],
                ):
                    system, opponent, system_name, opponent_name, played_as, end = view
                    if wanted is not None and system not in wanted:
                        continue
                    flipped = played_as == "team_b"
                    fact: Row = {
                        "system_id": system,
                        "system_name": system_name,
                        "opponent_id": opponent,
                        "opponent_name": opponent_name,
                        "played_as": played_as,
                        "spawn_end": end,
                        **{
                            name: identity[
                                _IDENTITY_TWINS.get(name, name) if flipped else name
                            ]
                            for name in IDENTITY_COLUMNS
                        },
                    }
                    labels = game_members.get(record["origin"], unlabeled)
                    for slot in range(10):
                        fact[f"agent_{slot}_member"] = labels[
                            (slot + 5) % 10 if flipped else slot
                        ]
                    for column, slots in (
                        ("member", range(5)),
                        ("opponent_member", range(5, 10)),
                    ):
                        fact[column] = _side_member(
                            [fact[f"agent_{slot}_member"] for slot in slots],
                            [fact[f"agent_{slot}_active"] for slot in slots],
                        )
                    fact_rows.append(fact)
                    positions_list.append(position)
                    mirrored_list.append(flipped)
            if not fact_rows:
                return {}
            result = _columns(
                fact_rows, table=table, current=self.manifest["schema_version"] == 2
            )
            positions = np.asarray(positions_list, np.int64)
            mirrored = np.asarray(mirrored_list, np.bool_)
            # Fill in column batches so temporary copies stay small.
            block = np.empty((len(positions), len(requested)), np.float32)
            for start in range(0, len(requested), _COLUMN_BATCH):
                stop = min(start + _COLUMN_BATCH, len(requested))
                block[:, start:stop] = oriented(
                    requested[start:stop], positions, mirrored, np.float32
                )
            for place, name in enumerate(requested):
                result[name] = block[:, place]
            for name, (top, bottom) in ratios.items():
                parts = oriented([top, bottom], positions, mirrored)
                with np.errstate(divide="ignore", invalid="ignore"):
                    result[name] = np.where(
                        parts[:, 1] > 0, parts[:, 0] / parts[:, 1], np.nan
                    )
            return result

        rankings: dict[str, Row] = (
            {
                str(row.get("system_id")): row
                for row in self.iter_rows("tournament_rankings", 256)
            }
            if self.metadata["tables"]["tournament_rankings"]["availability"]
            == "available"
            else {}
        )
        groups: dict[tuple[object, ...], Row] = {}

        def group_of(
            record: Row, system: str, opponent: str, names: tuple[str, str]
        ) -> Row:
            """Return the summary group of one System's view of one game."""
            key: tuple[object, ...] = (
                (system,)
                if by == "system"
                else (system, opponent)
                if by == "opponent"
                else (system, record["map_key"])
            )
            if key not in groups:
                facts: Row = {"system_id": system, "system_name": names[0]}
                if by == "opponent":
                    facts.update(opponent_id=opponent, opponent_name=names[1])
                elif by == "map":
                    facts.update(
                        map_id=record["map_id"],
                        config_id=record["config_id"]
                        if record["map_id"] is None
                        else None,
                    )
                groups[key] = {"facts": facts, "records": [], "rows": []}
            return groups[key]

        for record in plan:
            for view in _directions(
                record["a"],
                record["b"],
                record["name_a"],
                record["name_b"],
                record["spawn"],
            ):
                system, opponent, system_name, opponent_name, played_as, _ = view
                if wanted is None or system in wanted:
                    group_of(record, system, opponent, (system_name, opponent_name))[
                        "records"
                    ].append((record, opponent))
        for position, (record, identity) in enumerate(games):
            for view in _directions(
                record["a"],
                record["b"],
                record["name_a"],
                record["name_b"],
                record["spawn"],
            ):
                system, opponent, system_name, opponent_name, played_as, _ = view
                if wanted is None or system in wanted:
                    group_of(record, system, opponent, (system_name, opponent_name))[
                        "rows"
                    ].append((position, played_as == "team_b", record, identity))

        outputs = [*requested, *ratios]
        value_sources: list[str | None] = [
            *requested,
            *(parts[0] for parts in ratios.values()),
        ]
        weight_sources: list[str | None] = [
            *(weight_of[name] for name in requested),
            *(parts[1] for parts in ratios.values()),
        ]
        is_ratio = np.asarray([False] * len(requested) + [True] * len(ratios))
        slots = [
            {slot for part in ratios.get(name, (name,)) for slot in _metric_slots(part)}
            for name in outputs
        ]
        label = "Full metrics" if table == "full_metrics" else "Priority metrics"
        fact_rows = []
        metric_values: list[NDArray[np.float64]] = []
        metric_lows: list[NDArray[np.float64]] = []
        metric_highs: list[NDArray[np.float64]] = []

        def row_order(key: tuple[object, ...]) -> tuple[object, ...]:
            """Sort rows as head_to_head does: System, then opponent or map."""
            if by != "map":
                return tuple(str(part) for part in key)
            facts = groups[key]["facts"]
            map_id = facts["map_id"]
            return (
                str(key[0]),
                (0, map_id, "")
                if map_id is not None
                else (1, 0, str(facts["config_id"])),
            )

        for key in sorted(groups, key=row_order):
            group = groups[key]
            records = group["records"]
            rows = group["rows"]
            scheduled = len(records)
            played = sum(record["row"] is not None for record, _ in records)
            reasons: list[str] = []
            if any(not record["pass_complete"] for record, _ in records):
                reasons.append("Requested pass is incomplete or failed")
            if any(not record["root_known"] for record, _ in records):
                reasons.append("Random root coordinates are unavailable")
            samplings = {record["sampling"] for record, _ in records}
            if "unknown" in samplings:
                reasons.append("Sampling mechanism unknown")
            elif "stochastic" not in samplings:
                reasons.append("No random sampling variation")
            if played < scheduled:
                reasons.append("Requested games are incomplete")
            partial = len(rows) < played
            if partial:
                reasons.append(
                    f"{label} cover only {len(rows)} of {played} played games "
                    "of this row"
                )
            recorded = {record["origin"] for _, _, record, _ in rows}
            pairs: dict[tuple[str, ...], list[tuple[Row, str]]] = {}
            for record, opponent in records:
                if record["pair"] is None:
                    reasons.append("Verified exchanged-spawn pairs are unavailable")
                    continue
                pairs.setdefault(record["pair"], []).append((record, opponent))
            units: dict[tuple[str, ...], list[tuple[Row, str]]] = {}
            seed_units: dict[tuple[object, ...], set[tuple[str, ...]]] = defaultdict(
                set
            )
            complete_pairs = 0
            for pair in pairs.values():
                first = pair[0][0]
                if (
                    len(pair) != 2
                    or {record["spawn"] for record, _ in pair} != {0, 1}
                    or len({record["cell"] for record, _ in pair}) != 1
                    or len({record["unit"] for record, _ in pair}) != 1
                    or len({record["seed"] for record, _ in pair}) != 1
                    or first["seed"][-1] is None
                    or any(record["origin"] not in recorded for record, _ in pair)
                ):
                    reasons.append(
                        "An exchanged-spawn pair is incomplete or inconsistent"
                    )
                    continue
                complete_pairs += 1
                units.setdefault(first["unit"], []).extend(pair)
                seed_units[first["seed"]].add(first["unit"])
            if any(len(value) > 1 for value in seed_units.values()):
                reasons.append(
                    "Repeated seed coordinates need one declared sampling block"
                )
            # Equal-coverage strata, as Elo uses: units covering the same
            # opponents and map cells, game for game, are drawn together.
            strata: dict[tuple[object, ...], list[tuple[str, ...]]] = defaultdict(list)
            for unit, members in units.items():
                signature = tuple(
                    sorted(
                        Counter(
                            (opponent, record["cell"]) for record, opponent in members
                        ).items()
                    )
                )
                strata[signature].append(unit)
            if not strata or any(len(value) < 2 for value in strata.values()):
                reasons.append(
                    "Fewer than two independent complete units in a condition stratum"
                )
            if complete_pairs * 2 != scheduled:
                reasons.append("Requested pair evidence is incomplete")
            reasons = list(dict.fromkeys(reasons))
            compute, status, reason, assumption = _interval_rule(reasons)
            stochastic_units = sum(
                any(record["sampling"] == "stochastic" for record, _ in members)
                for members in units.values()
            )

            positions = np.asarray([row[0] for row in rows], np.int64)
            mirrored = np.asarray([row[1] for row in rows], np.bool_)
            classes: list[object] = []
            unsettled: set[int] = set()
            for slot in range(10):
                seen = {
                    identity[f"agent_{(slot + 5) % 10 if flipped else slot}_class_id"]
                    for _, flipped, _, identity in rows
                }
                classes.append(next(iter(seen)) if len(seen) == 1 else None)
                if len(seen) > 1:
                    unsettled.add(slot)
            usable = np.asarray([not (slot_set & unsettled) for slot_set in slots])
            value = np.full(len(outputs), np.nan, np.float64)
            low = np.full(len(outputs), np.nan, np.float64)
            high = np.full(len(outputs), np.nan, np.float64)
            counts: NDArray[np.float64] | None = None
            by_unit = np.zeros(0, np.int64)
            unit_starts = np.zeros(0, np.int64)
            if rows and not partial and compute:
                ordered = sorted(strata.items())
                unit_position = {
                    unit: place
                    for place, unit in enumerate(
                        unit for _, members in ordered for unit in members
                    )
                }
                origin_unit = {
                    record["origin"]: unit_position[unit]
                    for unit, members in units.items()
                    for record, _ in members
                }
                unit_of = np.asarray(
                    [origin_unit[record["origin"]] for _, _, record, _ in rows],
                    np.int64,
                )
                by_unit = np.argsort(unit_of, kind="stable")
                unit_starts = np.flatnonzero(
                    np.r_[True, np.diff(unit_of[by_unit]) != 0]
                )
                chunks = _bootstrap_indices(
                    [len(members) for _, members in ordered], draws, seed
                )
                # Units are numbered stratum by stratum, so the strata's draw
                # counts sit side by side as one (draws, units) matrix.
                counts = np.concatenate(
                    [
                        _unit_counts(drawn, len(members))
                        for (_, members), drawn in zip(ordered, chunks, strict=True)
                    ],
                    axis=1,
                )
            if rows and not partial:
                for start in range(0, len(outputs), _COLUMN_BATCH):
                    stop = min(start + _COLUMN_BATCH, len(outputs))
                    top = oriented(value_sources[start:stop], positions, mirrored)
                    bottom = oriented(weight_sources[start:stop], positions, mirrored)
                    ratio = is_ratio[start:stop]
                    valid = (
                        np.isfinite(top) & np.isfinite(bottom) & (ratio | (bottom > 0))
                    )
                    numerator = np.where(valid, np.where(ratio, top, top * bottom), 0.0)
                    denominator = np.where(valid, bottom, 0.0)
                    total = denominator.sum(axis=0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        value[start:stop] = np.where(
                            total > 0, numerator.sum(axis=0) / total, np.nan
                        )
                    if counts is None:
                        continue
                    # Rows sorted by unit: each unit's games sum in one pass.
                    top_units = np.add.reduceat(numerator[by_unit], unit_starts, axis=0)
                    bottom_units = np.add.reduceat(
                        denominator[by_unit], unit_starts, axis=0
                    )
                    sampled_top = counts @ top_units
                    sampled_bottom = counts @ bottom_units
                    empty = np.any(sampled_bottom <= 0, axis=0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sampled = sampled_top / sampled_bottom
                    sampled[:, empty] = 0.0
                    bounds = np.percentile(
                        sampled, (2.5, 97.5), axis=0, method="linear"
                    )
                    bounds[:, empty] = np.nan
                    low[start:stop], high[start:stop] = bounds[0], bounds[1]
            value[~usable] = np.nan
            low[~usable | ~np.isfinite(value)] = np.nan
            high[~usable | ~np.isfinite(value)] = np.nan
            metric_values.append(value)
            metric_lows.append(low)
            metric_highs.append(high)
            ranking = rankings.get(str(group["facts"]["system_id"]), {})
            fact_rows.append(
                {
                    **group["facts"],
                    "games": len(rows),
                    "scheduled_games": scheduled,
                    "completed_pairs": complete_pairs,
                    "elo": ranking.get("elo", np.nan),
                    "elo_ci_low": ranking.get("elo_ci_low", np.nan),
                    "elo_ci_high": ranking.get("elo_ci_high", np.nan),
                    "rank": ranking.get("rank"),
                    **{f"agent_{slot}_class_id": classes[slot] for slot in range(10)},
                    "confidence": 0.95,
                    "independent_units": 0
                    if "No random sampling variation" in reasons
                    else None
                    if any(
                        reason in reasons
                        for reason in (
                            "Sampling mechanism unknown",
                            "Random root coordinates are unavailable",
                            "Repeated seed coordinates need "
                            "one declared sampling block",
                        )
                    )
                    else stochastic_units,
                    "interval_status": status,
                    "unavailable_reason": reason,
                    "conditional_interval_assumption": assumption,
                    "sampling_unit": "Whole declared blocks of exchanged-spawn pairs",
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed,
                }
            )
        if not fact_rows:
            return {}
        facts = _columns(fact_rows, table="system_metrics", current=False)
        split = list(facts).index("confidence")
        result = {name: facts[name] for name in list(facts)[:split]}
        stacked = (
            np.stack(metric_values),
            np.stack(metric_lows),
            np.stack(metric_highs),
        )
        for place, name in enumerate(outputs):
            result[name] = stacked[0][:, place]
            result[f"{name}_ci_low"] = stacked[1][:, place]
            result[f"{name}_ci_high"] = stacked[2][:, place]
        result.update({name: facts[name] for name in list(facts)[split:]})
        return result

    def _settings_entries(self) -> list[Row]:
        """Return the pass entries whose games this scope reports."""
        return [
            entry
            for entry in self.selected.values()
            if entry.get("completed_episode_ids")
        ]

    def execution_settings(self) -> dict[str, object]:
        """Merge the execution_settings of every pass with games, once.

        A pass that played games without a record counts as unknown. The
        merged record is cached on the view; callers receive a copy.
        """
        cached = cast(dict[str, object] | None, getattr(self, "_settings_cache", None))
        if cached is None:
            from marl_battlegrounds.evaluation.runtime_provenance import (
                merge_execution_settings,
            )

            records: list[Mapping[str, object] | None] = []
            for entry in self._settings_entries():
                saved = _mapping(entry.get("details")).get("execution_settings")
                records.append(
                    cast(Mapping[str, object], saved)
                    if isinstance(saved, Mapping)
                    else None
                )
            cached = merge_execution_settings(records)
            self._settings_cache = cached
        return {
            "declared_xla_flags_by_attempt": list(
                cast(list[object], cached["declared_xla_flags_by_attempt"])
            ),
            "declared_gpu_deterministic_ops": cached["declared_gpu_deterministic_ops"],
        }

    def _rankings(self, rows: int) -> Iterator[Row]:
        """Add exact competition ranks from saved unrounded Elo, without fitting."""
        values = list(self._raw("tournament_results.csv", rows))
        participants = _mapping(
            _mapping(self.manifest.get("details")).get("participants")
        )
        if not participants:
            for entry in self.selected.values():
                ids = _mapping(entry.get("system_ids"))
                for team, descriptor in _mapping(entry.get("policies")).items():
                    label = _mapping(descriptor).get("name")
                    if label is not None and ids.get(team) is not None:
                        previous = participants.get(str(label), ids[team])
                        participants[str(label)] = (
                            ids[team] if previous == ids[team] else None
                        )
        for row in values:
            label = row.get("policy")
            row["system_id"] = participants.get(str(label))
            row["system_name"] = label
        values.sort(
            key=lambda row: (
                row.get("elo") is None,
                -float(row.get("elo") or 0),
                str(row.get("system_id") or row.get("policy") or ""),
            )
        )
        previous: object = object()
        rank = 0
        for index, row in enumerate(values, start=1):
            rating = row.get("elo")
            if rating is not None and rating != previous:
                rank = index
            row["rank"] = None if rating is None else rank
            previous = rating
            yield row

    def iter_rows(
        self, name: str, rows: int, columns: frozenset[str] | None = None
    ) -> Iterator[Row]:
        """Apply one small projection over raw rows; never refit or reduce metrics.

        columns, used only for the two metric tables, keeps just those cells;
        it must include the identity columns. None keeps every cell.
        """
        description = self.metadata["tables"][name]
        availability = description["availability"]
        if availability == "disabled":
            return
        if availability != "available":
            raise ValueError(description["reason"])
        if name == "episodes":
            yield from self._episodes(rows)
            return
        if name == "policy_assignments":
            labels: dict[str, list[str]] = {}
            for row in self._raw("policy_assignments.csv", rows):
                system = str(row["system_id"])
                registration = self._registration(row)
                if system not in labels:
                    labels[system] = _member_labels(registration) or [
                        str(_mapping(item).get("name", ""))
                        for item in cast(
                            list[object], registration.get("components", [])
                        )
                    ]
                choice = int(row["policy_id"])
                row["system_name"] = registration.get("name")
                row["member"] = None if choice < 0 else labels[system][choice]
                yield row
            return
        if name == "priority_metrics":
            yield from self._raw("priority_metrics.csv", rows, columns)
            for row in self._raw("match_results.csv", rows, columns):
                entry = self._entry(row)
                coverage = _mapping(entry.get("recorded_metrics_by_episode"))
                measured = coverage.get(str(row["episode_id"])) in {"priority", "full"}
                if not coverage and self.manifest["schema_version"] == 1:
                    details = _mapping(entry.get("details"))
                    measured = details.get("metrics") in {
                        "priority",
                        "full",
                    } or row["episode_id"] in details.get("full_metrics_episodes", ())
                if measured:
                    yield {
                        name: row.get(name)
                        for name in description["columns"]
                        if columns is None or name in columns
                    }
            return
        iterator = (
            self._rankings(rows)
            if name == "tournament_rankings"
            else self._raw(
                _FILES[name], rows, columns if name == "full_metrics" else None
            )
        )
        for row in iterator:
            if name in _SUMMARIES:
                row.setdefault("run_id", self.manifest.get("run_id"))
                row.setdefault("phase", "tournament")
                row.setdefault("pass_id", None)
            yield row

    def _spawn_balance(self) -> object:
        """Count only completed recorded games and steps under their own bindings."""
        records: dict[tuple[str, str], Row] = {}
        lengths: dict[tuple[str, str, str], int] = {}
        for filename in ("episodes.csv", "match_results.csv"):
            for row in self._raw(filename, 128):
                if row.get("episode_length") is not None:
                    lengths[
                        (str(row["phase"]), str(row["pass_id"]), str(row["episode_id"]))
                    ] = int(row["episode_length"])
        for entry in self.selected.values():
            if entry.get(
                "pass_role"
            ) == "tournament_coordinator" or _historical_coordinator(
                entry, self.manifest
            ):
                continue
            details = _mapping(entry.get("details"))
            contract = _mapping(details.get("evaluation_contract"))
            schedule = _mapping(entry.get("episodes"))
            completed = set(map(str, entry.get("completed_episode_ids", ())))
            games = dict.fromkeys(("default", "swapped", "unreported"), 0)
            steps: dict[str, int | None] = {name: 0 for name in games}
            missing_lengths: list[int] = []
            pairs: dict[str, list[tuple[str, object]]] = {}
            for identifier, declaration in schedule.items():
                declaration = _mapping(declaration)
                choice = declaration.get("spawn_locations")
                key = declaration.get(
                    "paired_comparison_key", declaration.get("comparison_key")
                )
                kind = declaration.get("comparison_kind")
                if key is not None and kind == "verified_spawn_pair":
                    pairs.setdefault(str(key), []).append((identifier, choice))
                if identifier in completed:
                    label = (
                        "default"
                        if choice == 0
                        else "swapped"
                        if choice == 1
                        else "unreported"
                    )
                    games[label] += 1
                    length = lengths.get((entry["phase"], entry["pass_id"], identifier))
                    if length is None:
                        steps[label] = None
                        missing_lengths.append(int(identifier))
                    elif steps[label] is not None:
                        steps[label] = cast(int, steps[label]) + length
            for identifier in completed - set(schedule):
                games["unreported"] += 1
                length = lengths.get((entry["phase"], entry["pass_id"], identifier))
                if length is None:
                    steps["unreported"] = None
                    missing_lengths.append(int(identifier))
                elif steps["unreported"] is not None:
                    steps["unreported"] += length
            complete_pairs = sum(
                len(group) == 2
                and {value for _, value in group} == {0, 1}
                and all(identifier in completed for identifier, _ in group)
                for group in pairs.values()
            )
            ids = _mapping(entry.get("system_ids"))
            mode = contract.get(
                "spawn_mode",
                contract.get(
                    "requested_spawn_locations",
                    details.get("spawn_locations", "explicit"),
                ),
            )
            declared = (
                bool(pairs)
                or mode == "paired"
                or contract.get("pairing_protocol") is not None
            )
            paired = (
                None
                if not declared
                else bool(schedule)
                and complete_pairs * 2 == len(schedule)
                and len(completed) == len(schedule)
            )
            records[(entry["phase"], entry["pass_id"])] = {
                "mode": mode,
                "system_id": ids.get("team_a"),
                "opponent_id": ids.get("team_b"),
                "completed_games": games,
                "completed_steps": steps,
                "expected_pairs": len(schedule) // 2
                if mode == "paired"
                else len(pairs),
                "completed_pairs": complete_pairs,
                "missing_episode_ids": tuple(
                    sorted(map(int, set(schedule) - completed))
                ),
                "paired_complete": paired,
                "missing_episode_lengths": tuple(sorted(missing_lengths)),
                "source_coverage": _source_coverage(entry, lengths),
            }
        return next(iter(records.values())) if len(records) == 1 else records

    @property
    def replay_paths(self) -> tuple[Path, ...]:
        """Return existing committed replays in the recorded episode order.

        Prefer each pass's explicit contract order, then its older declared ID
        list. Whole-tournament views use the shared global schedule. Historical
        files without an ordered declaration fall back to ascending episode IDs;
        JSON object key order never claims an execution or completion order.
        """
        if self.run_dir is None:
            return ()
        result: list[tuple[str, str, Path]] = []
        for entry in self.selected.values():
            records = _mapping(entry.get("replays"))
            schedule = _mapping(entry.get("episodes"))
            details = _mapping(entry.get("details"))
            contract = _mapping(details.get("evaluation_contract"))
            declared = tuple(
                map(str, contract.get("episode_ids", details.get("episode_ids", ())))
            )
            if len(set(declared)) != len(declared):
                raise ValueError("recorded replay schedule repeats an episode ID")
            remaining = sorted((set(schedule) | set(records)) - set(declared), key=int)
            for episode_id in (*declared, *remaining):
                record = records.get(episode_id)
                if record is None:
                    continue
                relative = Path(record["path"])
                if len(relative.parts) != 2 or relative.parts[0] != "replays":
                    raise ValueError("committed replay path leaves the run")
                path = self.run_dir / relative
                if (
                    path.is_symlink()
                    or path.parent.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != record["bytes"]
                ):
                    raise ValueError("committed replay is missing, changed or unsafe")
                result.append((entry["phase"], episode_id, path))
        if self.whole_tournament:
            schedule = _mapping(self.manifest.get("details")).get("schedule", ())
            order = {
                str(row["episode_id"]): index for index, row in enumerate(schedule)
            }
            tournament = iter(
                sorted(
                    (row for row in result if row[0] == "tournament"),
                    key=lambda row: (order.get(row[1], len(order)), int(row[1])),
                )
            )
            return tuple(
                next(tournament)[2] if phase == "tournament" else path
                for phase, _, path in result
            )
        return tuple(path for _, _, path in result)


def _result_view(
    manifest: Row,
    *,
    run_dir: Path | None,
    phase: str | None = None,
    pass_id: str | None = None,
    memory: Mapping[str, object] | None = None,
) -> _View:
    """Dispatch additive canonical references without importing execution code."""
    if "tournament_reuse" in manifest:
        from marl_battlegrounds.evaluation.canonical_results import CanonicalView

        return CanonicalView(
            manifest, run_dir=run_dir, phase=phase, pass_id=pass_id, memory=memory
        )
    return _View(manifest, run_dir=run_dir, phase=phase, pass_id=pass_id, memory=memory)


class _ResultAccess:
    """Share bounded table methods across saved, evaluation and tournament results."""

    _view: _View

    @property
    def status(self) -> str:
        """Return complete, incomplete or failed for the exact selected scope."""
        return self._view.status

    @property
    def run_dir(self) -> Path | None:
        """Return the exact saved run directory, or None for in-memory results."""
        return self._view.run_dir

    @property
    def replay_paths(self) -> tuple[Path, ...]:
        """Return existing committed replay paths, or () for memory-only replays.

        Paths follow the recorded schedule, including global tournament order.
        Older files without an ordered declaration use ascending episode IDs;
        that fallback does not claim the original execution or completion order.
        Missing, changed or unsafe committed replay files raise ValueError.
        """
        return self._view.replay_paths

    @property
    def execution_settings(self) -> dict[str, object]:
        """Say whether XLA's deterministic GPU mode was declared for these games.

        Returns
        -------
        dict
            ``declared_gpu_deterministic_ops``: True when every attempt of
            every pass with games declared ``--xla_gpu_deterministic_ops``,
            False when any declared it off or left it out, and None when any
            attempt saved no record (runs made before this record existed) or
            used an unknown spelling. ``declared_xla_flags_by_attempt``: each
            attempt's whole XLA_FLAGS string in pass order, with None for an
            attempt without a record. Canonical results include the source
            passes of reused games.

        Notes
        -----
        Exact repeats of saved games on a GPU need deterministic mode, the same
        GPU model and the same software (``metadata["runtime_provenance"]``).
        This reports what was declared, not what XLA parsed. Computed on first
        access and cached; canonical sources load only then.
        """
        return self._view.execution_settings()

    def iter_table(self, name: str, rows: int = 128) -> Iterator[Columns]:
        """Yield at most ``rows`` recorded rows as NumPy columns per batch.

        Parameters
        ----------
        name : str
            Name listed in ``metadata['tables']``. Raw views retain completion
            order. Summary views require the whole completed tournament scope.
        rows : int, optional
            Positive batch size, default 128. Booleans are not valid sizes.

        Yields
        ------
        dict[str, numpy.ndarray]
            One bounded host batch. Missing metric cells are NaN; nullable
            identity/count cells use None. No file is written or result fitted.

        Raises
        ------
        ValueError
            Name, size, availability, schema or a saved boundary is invalid.
        OSError
            A saved file cannot be read. Disabled/available-empty tables yield
            nothing; their metadata distinguishes these cases.
        """
        if name not in _TABLES:
            raise ValueError(
                f"unknown table {name!r}; choose from {', '.join(_TABLES)}"
            )
        if type(rows) is not int or rows <= 0:
            raise ValueError("rows must be a positive integer, excluding booleans")
        batch: list[Row] = []
        for row in self._view.iter_rows(name, rows):
            batch.append(row)
            if len(batch) == rows:
                yield _columns(
                    batch,
                    table=name,
                    current=self._view.manifest["schema_version"] == 2,
                )
                batch.clear()
        if batch:
            yield _columns(
                batch, table=name, current=self._view.manifest["schema_version"] == 2
            )

    def table(self, name: str) -> Columns:
        """Allocate the whole selected table as NumPy columns.

        ``name`` uses the same names, scope, errors and missing-value rules as
        ``iter_table``. This explicit convenience can use substantial RAM for
        full reports. Disabled or available-empty tables return an empty dict.
        Existing saved files and legacy fields are never changed.
        """
        batches = list(self.iter_table(name))
        if not batches:
            return {}
        if len(batches) == 1:
            return batches[0]
        return {
            name: np.concatenate([batch[name] for batch in batches])
            for name in batches[0]
        }

    def head_to_head(self, by: str = "overall") -> Columns:
        """Summarize completed games against each opponent using exact identities.

        Parameters
        ----------
        by : {"overall", "map", "spawn"}, default="overall"
            Keep one directed row per System/opponent, or split it by map or
            the focal System's physical spawn bank. spawn_end is team_a for
            the map's original Team A bank and team_b for its original Team B
            bank. Exchanging ends never exchanges the recorded teams. Custom
            maps without map_id stay separated by their saved config_id.

        Returns
        -------
        dict[str, numpy.ndarray]
            NumPy columns with exact IDs, display names, games, W/D/L, rates,
            mean_points_for, mean_points_against and mean_point_margin. Each
            completed game has equal weight. Both directions appear; identical
            System self-play appears once from recorded Team A's view. Counts
            remain exact integers. Missing points/rates are NaN; unavailable
            bounds are NaN with an explicit status. A complete tournament
            matchup can reuse its saved expected-score interval. Point-margin intervals
            are not estimated. Empty completed scope returns {}.

        Raises
        ------
        ValueError
            by is unknown, original System identities are missing, a requested
            spawn end is missing, or the saved snapshot fails its normal checks.

        Notes
        -----
        Available with metrics="none": required outcomes and native points
        are always recorded. Optional combat metrics are never reconstructed.
        Host-only; reads narrow outcomes once per call, without JAX, rating fits
        or saved-file changes. Canonical results retain declared entrant IDs;
        other results retain recorded System registration IDs. Existing raw
        tables preserve every game's original run/pass/episode ownership. A
        reused expected-score interval keeps its saved scope and assumptions;
        read matchup_results for those original sampling facts. It supplies no
        uncertainty estimate for point margin.
        """
        if by not in ("overall", "map", "spawn"):
            raise ValueError("by must be overall, map or spawn")
        return _columns(
            self._view.head_to_head(by), table="head_to_head", current=False
        )

    def point_margin_summary(
        self,
        *,
        by: str = "overall",
        bootstrap_draws: int = 2000,
        bootstrap_seed: int = 19044001,
    ) -> Columns:
        """Report native Team A point margins and fixed-method game intervals.

        Parameters
        ----------
        by : {"overall", "map"}, default="overall"
            One row per recorded Team A/System and Team B/opponent identity,
            optionally split by map. Spawn exchange does not reverse Team A.
            Every completed game has equal weight, including across maps.
        bootstrap_draws : int, default=2000
            At least two whole-unit resampling draws. Booleans are rejected.
        bootstrap_seed : int, default=19044001
            Integer seed from 0 through 2**63 - 1 for a private NumPy generator.
            Global random state is unchanged. Booleans are rejected.

        Returns
        -------
        dict[str, numpy.ndarray]
            Exact identities, planned/completed games and pairs, mean native
            Team A minus Team B points, and 95% percentile interval columns.
            Bounds use linear 2.5/97.5 percentiles. Whole verified spawn pairs
            stay together; declared larger blocks use one draw across linked
            maps. Each condition stratum needs two independent complete units.
            Incomplete games/pairs, deterministic repetition (no stochastic
            game in the row) or missing points keep descriptive rows but yield
            NaN bounds, ``interval_status``
            "Unavailable" and an ``unavailable_reason``. When the only problem
            is a method whose sampling is unknown (such as your own System),
            the bounds are kept, ``interval_status`` is "Conditional: Sampling
            not verified" (also when that row mixes unknown and deterministic
            passes, as ``system_metrics`` decides) and
            ``conditional_interval_assumption`` states what these bounds
            assume. Elo rankings use the same label, but their assumption
            column describes separate ``conditional_*`` columns. Otherwise
            that column is None.
            Empty declared scope returns {}. These intervals describe fixed
            methods, not uncertainty across training seeds.

        Raises
        ------
        ValueError
            Arguments, saved identities, sampling facts or snapshots are invalid.
        OSError
            A saved outcome file cannot be read.

        Notes
        -----
        Works with metrics="none". Reads narrow outcomes only, on the host,
        without JAX, policy calls or file writes. Memory holds narrow game rows
        and the saved schedule; bootstrap draw batches are bounded. Calling
        head_to_head does not perform this resampling work. A canonical result
        that reuses another run's games includes those games, with their
        original declarations; pair keys carry each game's full origin (run,
        phase, pass and pair), so a reused pair never merges with a local pair
        that has the same labels. ``system_metrics(by="opponent")`` gives the
        same ``score_difference`` interval for recorded Team A rows.
        """
        if by not in ("overall", "map"):
            raise ValueError("by must be overall or map")
        if type(bootstrap_draws) is not int or bootstrap_draws < 2:
            raise ValueError("bootstrap_draws must be an integer of at least two")
        if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed <= 2**63 - 1:
            raise ValueError(
                "bootstrap_seed must be an integer from 0 through 2**63 - 1"
            )
        return _columns(
            self._view.point_margin_summary(by, bootstrap_draws, bootstrap_seed),
            table="point_margin_summary",
            current=False,
        )

    def system_metrics(
        self,
        table: str,
        *,
        by: str = "system",
        systems: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
        ratios: Mapping[str, tuple[str, str]] | None = None,
        bootstrap_draws: int = 2000,
        bootstrap_seed: int = 19044001,
    ) -> Columns:
        """Show every System's metrics from its own side, ready to compare.

        Each System is shown as Team A and its opponent as Team B, whichever
        side it really played, so ``team_a_kills`` is always that System's
        kills, ``agent_0`` to ``agent_4`` are its agents, and
        ``score_difference`` is its points minus its opponent's. Column names
        do not change. A game a System played as Team B is read from each
        column's mirrored twin (``metric_catalog.mirror_metric``).

        Parameters
        ----------
        table : {"priority_metrics", "full_metrics"}
            Metric table to read, by the names ``table()`` uses.
        by : {"system", "opponent", "map", "game"}, default="system"
            Row grain: one row per System; per System and opponent (the same
            rows as ``head_to_head()``, both directions, self-play once); per
            System and map (all its opponents on that map); or per System per
            game. Unlike head_to_head, ``by`` names the grain, not an extra
            split on top of System and opponent.
        systems : iterable of str or None, default=None
            System IDs or display names to keep; None keeps every System. Any
            non-empty, non-string iterable works, such as ``field["system_id"]``
            from an earlier call. A display name that two IDs share raises an
            error naming both. A single string is not accepted; use a one-item
            list.
        columns : iterable of str or None, default=None
            Metric column names to report, as the table names them; None reports
            every metric column in the table. An empty list is an error. Fewer
            columns use less memory and summary time; reading still splits
            each saved row once.
        ratios : mapping of str to (str, str) or None, default=None
            Extra combined fractions: new name -> (numerator column, denominator
            column), both read from the System's side. A summary gives total
            numerator / total denominator over games where both are recorded,
            with intervals, and NaN when that total denominator is not
            positive; ``by="game"`` gives each game's ratio (NaN when the
            denominator is not positive). Use a denominator that cannot be
            negative. Their input columns are read even when
            ``columns`` leaves them out. Example: ``{"slowed_kill_fraction":
            ("team_a_kills_of_hunter_basic_slow_recipient", "team_a_kills")}``.
        bootstrap_draws : int, default=2000
            At least two whole-unit resampling draws; booleans are rejected.
            Not used with ``by="game"``.
        bootstrap_seed : int, default=19044001
            Seed from 0 through 2**63 - 1 for a private NumPy generator, the
            same rule and draws as ``point_margin_summary``. Not used with
            ``by="game"``.

        Returns
        -------
        dict of str to numpy.ndarray
            ``by="game"``: ``system_id``, ``system_name``, ``opponent_id``,
            ``opponent_name``, ``played_as`` ("team_a" or "team_b", the side the
            System really played), ``spawn_end`` (the map's original spawn bank
            it started from, as in ``head_to_head(by="spawn")``, or None when
            not saved), the identity columns from the System's side,
            ``agent_<slot>_member`` for slots 0-9 from the System's side (the
            member that controlled the slot for the whole game, None otherwise),
            ``member`` and ``opponent_member`` (each side's distinct member
            labels joined with " / " in slot order, None when a member was not
            recorded or that System has none; see ``table("policy_assignments")``),
            each metric column (float32, NaN when unavailable) and each ratio
            (float64). Rows follow the table's order, Team A's view first.

            Summaries: ``system_id`` and ``system_name`` (plus ``opponent_id``
            and ``opponent_name``, or ``map_id`` and ``config_id``), ``games``
            (games with rows in the table), ``scheduled_games``,
            ``completed_pairs``, the System's own ``elo``, ``elo_ci_low``,
            ``elo_ci_high`` and ``rank`` copied from the saved rankings (NaN or
            None outside a complete tournament), ``agent_<slot>_class_id`` (the
            slot's class when it never changed, else None), then each metric
            and ratio under its own name with ``<name>_ci_low`` and
            ``<name>_ci_high``, then ``confidence``, ``independent_units``,
            ``interval_status`` ("Available", "Conditional: Sampling not
            verified" or "Unavailable"), ``unavailable_reason``,
            ``conditional_interval_assumption``, ``sampling_unit``,
            ``bootstrap_draws`` and ``bootstrap_seed`` as in
            ``point_margin_summary``, except that ``independent_units``
            counts only resampled units from matchups with random sampling.
            Values are float64. Rows are sorted by System ID, then by opponent
            ID or by map ID (configurations without a map ID last), as in
            ``head_to_head``.

            A disabled table, or no System to show, returns {}.

        Raises
        ------
        ValueError
            An argument is invalid, a column or System is unknown, a name is
            ambiguous, a ratio name is already a column name, or the table is
            unavailable in this scope.
        OSError
            A saved file cannot be read.

        Notes
        -----
        **How games combine.** Each column's ``summary_weight`` in the catalog
        decides: a fraction or rate is total numerator / total denominator, an
        average is weighted by its count of observations, and counts, health
        and ticks are means per game. Listed fractions without a saved
        denominator (``metric_catalog.SUMMARY_WEIGHT_MISSING``) are means per
        game. A game whose value is missing, or whose weight is zero, adds
        nothing to that column; a column with no usable game is NaN.

        **Sampling facts.** Intervals need to know how each method chooses its
        actions. Built-in methods and saved actors from the built-in learners
        declare it; other Systems are "unknown". Every summary row with one of
        their games keeps its interval but labels it "Conditional: Sampling not
        verified", the label Elo rankings give unknown sampling: the interval
        assumes the game pairs are independent repeats, which the records
        cannot prove (a System that carries memory from one game to the next
        breaks that). In a canonical run with your own challenger, that
        includes each member's overall row. Member-against-member
        ``by="opponent"`` rows keep their own label: "Available" when either
        member samples at random, "Unavailable" with "No random sampling
        variation" when neither does (scripted bots, greedy QMIX or PQN-VDN).
        ``interval_status`` covers a row's metric columns only; the copied
        ``elo_ci_*`` bounds keep the rankings' own labels.

        **In-memory evaluations.** An ``evaluate`` call without ``output_dir``
        inspects how its methods sample, as saved runs do, so its summaries
        carry the same interval labels. In-memory tournaments do too.

        **Missing data.** If games were played but some have no row in this
        table (full metrics recorded for only some games), a summary row's
        values are NaN and ``unavailable_reason`` says how many are covered;
        ``by="opponent"`` rows with every game covered stay valid, and
        ``by="game"`` still shows the recorded games. Games not played yet
        keep descriptive values but no interval. When a slot's class changes
        between a System's games, that slot's agent, pair and recipient
        columns are NaN in summaries.

        **Intervals.** Whole verified exchanged-spawn pairs, or declared
        blocks, are drawn with replacement inside equal-coverage groups of
        opponent and map, as for Elo; each column is recomputed from the drawn
        totals and the 2.5th and 97.5th percentiles (linear) are reported. A
        row withholds its interval with a reason in the cases
        ``point_margin_summary`` does, and also when coverage is partial or no
        drawn unit can vary ("No random sampling variation"). Matchups where
        both sides choose without randomness add no width to a mixed row. A
        column's bounds are NaN when some draw has no usable game for it, even
        when the row's status is "Available" or conditional. These intervals
        describe repeated games of fixed methods, not training seeds. With
        thousands of columns, many intervals exclude a difference by chance,
        and overlapping intervals between two Systems are not a test.
        Separate ``evaluate`` calls combined into one result need different
        seeds.

        **Cost.** Host-only NumPy: no JAX, no file writes. One pass over the
        table converts only the needed columns (requested, ratio inputs, their
        mirrored twins and weights) into float32 rows; summaries then work one
        group at a time in batches of columns. In-memory results convert every
        cell of their borrowed rows, so the projection saves time only for saved
        runs. ``by="game"`` on every full metric of a 7,200-game tournament
        returns about 645 MB; use ``columns`` and ``systems`` to keep it small.

        Examples
        --------
        >>> import marl_battlegrounds as marl_bgs
        >>> results = marl_bgs.load_results("runs/my_tournament")  # doctest: +SKIP
        >>> field = results.system_metrics("full_metrics")  # doctest: +SKIP
        >>> field["team_a_ally_distance_mean"], field["elo"]  # doctest: +SKIP
        """
        if table not in ("priority_metrics", "full_metrics"):
            raise ValueError("table must be priority_metrics or full_metrics")
        if by not in ("system", "opponent", "map", "game"):
            raise ValueError("by must be system, opponent, map or game")
        picked: dict[str, list[str] | None] = {}
        for name, value in (("systems", systems), ("columns", columns)):
            # Any non-string iterable works, such as a column of an earlier result.
            items = (
                None
                if value is None
                else list(cast(Iterable[object], value))
                if not isinstance(value, str)
                and isinstance(cast(object, value), Iterable)
                else [cast(object, value)]
            )
            if items is not None and (
                isinstance(value, str)
                or not items
                or not all(isinstance(item, str) and item for item in items)
            ):
                raise ValueError(
                    f"{name} must be a nonempty list of nonempty strings, or None"
                )
            picked[name] = None if items is None else [str(item) for item in items]
        if picked["columns"] is not None and len(set(picked["columns"])) != len(
            picked["columns"]
        ):
            raise ValueError("columns must not repeat a name")
        checked: dict[str, tuple[str, str]] = {}
        if ratios is not None:
            if not isinstance(cast(object, ratios), Mapping):
                raise ValueError("ratios must map names to (numerator, denominator)")
            for name, parts in ratios.items():
                if (
                    not isinstance(cast(object, name), str)
                    or not name
                    or isinstance(parts, str)
                    or not isinstance(cast(object, parts), Sequence)
                    or len(parts) != 2
                    or not all(
                        isinstance(cast(object, part), str) and part for part in parts
                    )
                ):
                    raise ValueError(
                        "ratios must map names to (numerator, denominator) column names"
                    )
                checked[name] = (parts[0], parts[1])
        if type(bootstrap_draws) is not int or bootstrap_draws < 2:
            raise ValueError("bootstrap_draws must be an integer of at least two")
        if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed <= 2**63 - 1:
            raise ValueError(
                "bootstrap_seed must be an integer from 0 through 2**63 - 1"
            )
        return self._view.system_metrics(
            table,
            by,
            picked["systems"],
            picked["columns"],
            checked,
            bootstrap_draws,
            bootstrap_seed,
        )

    def opponent_matrix(self, metric: str = "mean_point_margin") -> dict[str, Any]:
        """Pivot overall head-to-head values with exact IDs on both axes.

        Parameters
        ----------
        metric : str, default="mean_point_margin"
            games, wins, draws, losses, win_rate, draw_rate, loss_rate,
            expected_score, mean_points_for, mean_points_against or
            mean_point_margin. It names a head_to_head column, not a new metric.

        Returns
        -------
        dict
            system_ids and system_names are ordered tuples; values is a
            float64 (N,N) NumPy array, row System against column opponent.
            metric records the chosen column. Unplayed cells, including an
            unplayed diagonal, are NaN, never zero. Recorded self-play remains
            present. Display names can repeat without merging identities.
            Float conversion applies to this display matrix only; exact counts
            remain in head_to_head. A scope without known Systems gives (0,0).

        Raises
        ------
        ValueError
            The metric is unsupported or head_to_head cannot read the evidence.

        Notes
        -----
        Uses the same narrow reduction as head_to_head. No rating or confidence
        interval is fitted, no optional report is loaded, and no file changes.
        """
        if metric not in {
            "games",
            "wins",
            "draws",
            "losses",
            "win_rate",
            "draw_rate",
            "loss_rate",
            "expected_score",
            "mean_points_for",
            "mean_points_against",
            "mean_point_margin",
        }:
            raise ValueError("Unsupported opponent matrix metric")
        rows = self._view.head_to_head("overall")
        names = self._view.participant_names()
        for row in rows:
            names[row["system_id"]] = row["system_name"]
            names[row["opponent_id"]] = row["opponent_name"]
        ids = tuple(sorted(names))
        positions = {identifier: index for index, identifier in enumerate(ids)}
        values = np.full((len(ids), len(ids)), np.nan, np.float64)
        for row in rows:
            value = row[metric]
            values[positions[row["system_id"]], positions[row["opponent_id"]]] = (
                np.nan if value is None else value
            )
        return {
            "system_ids": ids,
            "system_names": tuple(names[key] for key in ids),
            "metric": metric,
            "values": values,
        }


@dataclass(frozen=True)
class EpisodeResult:
    """Keep one exact required game outcome, independent of optional metrics.

    IDs and scores are Python integers; map_id can be None for a custom map.
    ``episode_length`` counts real transitions from this episode's start, even
    when the authored simulator tick starts above zero. ``config_id`` names the
    exact resolved configuration. Outcome is 1/2 for Team A/B wins or 3 for draw.
    """

    episode_id: int
    seed_id: int
    map_id: int | None
    outcome: int
    episode_length: int
    team_a_score: int
    team_b_score: int
    config_id: str


def _memory_manifest(
    metadata: Row, *, tournament: bool, completed: Sequence[int]
) -> Row:
    """Project already resolved host metadata into the common read-only schema."""
    passes: Any = metadata.get("passes") if tournament else None
    if not isinstance(passes, dict):
        phase, pass_id = (
            str(metadata.get("phase", "evaluation")),
            str(metadata.get("pass_id", "1")),
        )
        schedule = metadata.get("schedule", {})
        if isinstance(schedule, (tuple, list)):
            schedule = {
                str(row["episode_id"]): row for row in cast(Sequence[Row], schedule)
            }
        else:
            schedule = {str(key): value for key, value in _mapping(schedule).items()}
        passes = {
            f"{phase}:{pass_id}": {
                "phase": phase,
                "pass_id": pass_id,
                "details": dict(metadata),
                "episodes": schedule,
                "completed_episode_ids": list(completed),
                "policies": metadata.get("policies", {}),
                "system_ids": metadata.get("system_ids", {}),
                "replays": {},
                "recorded_metrics_by_episode": {
                    str(value): "full"
                    if metadata.get("metrics") == "full"
                    or value in metadata.get("full_metrics_episodes", ())
                    else metadata.get("metrics", "priority")
                    for value in completed
                },
                "result_state": {
                    "version": 1,
                    "status": metadata.get("status", "complete"),
                    "schedule_digest": metadata.get("schedule_digest", "memory"),
                    "reason": None,
                },
            }
        }
    result: Row = {
        "schema_version": 2,
        "metric_schema_id": METRIC_SCHEMA_ID,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "run_id": metadata.get("run_id"),
        "passes": passes,
        "systems": metadata.get("systems", {}),
        "configurations": metadata.get("configurations", {}),
        "tables": {},
        "details": dict(metadata),
    }
    if tournament and "tournament_reuse" in metadata:
        result["tournament_reuse"] = metadata["tournament_reuse"]
    if tournament:
        result["tournament_summary"] = metadata.get(
            "tournament_summary", {"digest": "in-memory-qualified"}
        )
    return result


@dataclass(frozen=True)
class EvaluationResult(_ResultAccess):
    """Expose evaluation output and scoped host tables without owning a writer.

    Legacy fields keep their existing meanings: saved metric dicts are empty,
    and ``episodes`` contains only newly completed games. Uniform table methods
    include the whole selected saved pass, including earlier resumed games.
    ``metadata`` records table availability, status, scope and spawn coverage.
    Memory arrays are borrowed, not copied into a second wide report.

    Attributes
    ----------
    priority_metrics, full_metrics : dict of str to numpy.ndarray
        Without a writer: column name to one-dimensional NumPy array, one row
        per selected game, in episode ID order. priority_metrics has every
        game, except with metrics="none", when it has only the
        full_metrics_episodes games. full_metrics has only the
        full_metrics_episodes games, except with metrics="full", when it has
        every game. Each dict starts with the IDENTITY_COLUMNS: run_id, phase,
        pass_id, episode_id (int32), seed_id (uint32), map_id, config_id,
        team_a_policy, team_b_policy, checkpoint_id, and each slot's
        agent_<slot>_class_id and agent_<slot>_active. The float32 metric
        columns follow, where NaN means unavailable. A table with no selected
        game is an empty dict. With a writer both are empty; the values stay
        in its files.
    episodes : tuple of EpisodeResult
        Compact required outcomes of the games completed by this call, in
        episode ID order.
    metadata : dict
        The actual schedule, systems, settings and run identity. Creating the
        result adds table availability, status, scope and spawn coverage to
        this same dictionary. A generated pass records its resolved Red Zone
        depth, in map units, as "red_zone_depth" in the options of
        ``metadata["evaluation_contract"]`` (0.0 means the rule is off).
    completed_episode_ids : tuple of int, default=()
        Every durably completed episode ID in this pass, earlier resumed
        games included, in ascending order. Empty means this call's
        ``episodes`` are the whole pass.
    paths : dict of str to Path or None, default=None
        The writer's files, by name. None for in-memory results.
    replays : tuple of ReplayArtifactV4, default=()
        Current (version 4) replays captured in memory by this call. Empty
        with a writer, which saves them as files instead.
    """

    priority_metrics: Columns
    full_metrics: Columns
    episodes: tuple[EpisodeResult, ...]
    metadata: Row
    completed_episode_ids: tuple[int, ...] = ()
    paths: dict[str, Path] | None = None
    replays: tuple[ReplayArtifactV4, ...] = ()
    _view: _View = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Bind saved scope or borrow memory columns and exact required outcomes."""
        if self.paths and "run_details" in self.paths:
            directory = Path(self.paths["run_details"]).parent
            view = _View(
                _read_manifest(directory),
                run_dir=directory,
                phase=self.metadata.get("phase"),
                pass_id=self.metadata.get("pass_id"),
            )
        else:
            completed = self.completed_episode_ids or tuple(
                value.episode_id for value in self.episodes
            )
            manifest = _memory_manifest(
                self.metadata, tournament=False, completed=completed
            )
            entry = next(iter(manifest["passes"].values()))
            by_id = {value.episode_id: value for value in self.episodes}
            order = self.metadata.get("completion_order", tuple(by_id))
            rows: list[Row] = []
            for identifier in order:
                value = by_id[identifier]
                declaration = _mapping(entry["episodes"].get(str(identifier)))
                row = {name: declaration.get(name) for name in IDENTITY_COLUMNS}
                row.update(
                    run_id=manifest["run_id"],
                    phase=entry["phase"],
                    pass_id=entry["pass_id"],
                    episode_id=value.episode_id,
                    seed_id=value.seed_id,
                    map_id=value.map_id,
                    config_id=value.config_id,
                    outcome=value.outcome,
                    episode_length=value.episode_length,
                    team_a_score=value.team_a_score,
                    team_b_score=value.team_b_score,
                )
                policies = self.metadata.get("policies", ())
                if (
                    isinstance(policies, (tuple, list))
                    and len(cast(Sequence[object], policies)) == 2
                ):
                    descriptions = cast(Sequence[object], policies)
                    row["team_a_policy"] = _mapping(descriptions[0]).get("name")
                    row["team_b_policy"] = _mapping(descriptions[1]).get("name")
                row["checkpoint_id"] = self.metadata.get("checkpoint_id")
                config = _mapping(
                    _mapping(manifest.get("configurations")).get(value.config_id)
                )
                profile = _mapping(config.get("agent_profile"))
                for slot in range(10):
                    for suffix, source in (
                        ("class_id", "class_ids"),
                        ("active", "active_mask"),
                    ):
                        if source in profile:
                            row[f"agent_{slot}_{suffix}"] = int(profile[source][slot])
                rows.append(row)
            memory: Row = {"episodes": tuple(rows), "completion_order": order}
            for name, columns in (
                ("priority_metrics", self.priority_metrics),
                ("full_metrics", self.full_metrics),
            ):
                if columns:
                    memory[name] = columns
            view = _View(manifest, run_dir=None, memory=memory)
        object.__setattr__(self, "_view", view)
        self.metadata.update(view.metadata)


@dataclass(frozen=True)
class TournamentResult(_ResultAccess):
    """Keep existing tournament fields plus uniform complete-population views.

    Row tuples are the already computed summary outputs. ``headline_metrics`` is
    an optional keyword-only table computed by the runner once. No accessor fits
    Elo or reconstructs missing measurements. Saved views use committed files;
    in-memory views borrow the supplied rows and wide full-metric arrays.

    Attributes
    ----------
    matches : tuple of dict
        One row per completed game: the match_results table with its outcome,
        scores and priority values.
    tournament_results : tuple of dict
        One row per entrant: centered rating, rates, interval availability and
        weighted worst-20-percent expected score.
    matchup_results : tuple of dict
        Directed policy/opponent rows with rates and score intervals.
    map_results : tuple of dict
        Policy/map rows with opponent-weighted rates and match counts.
    full_metrics : dict of str to numpy.ndarray
        Selected full metric columns captured in memory by this call. Empty
        with file output or when full metrics are off.
    replays : tuple of ReplayArtifactV4
        Current (version 4) replays captured in memory by this call. Empty with
        file output, which saves them as files instead.
    metadata : dict
        Method details, systems, participants, passes, configurations and the
        run's settings. Creating the result adds table availability, status,
        scope and spawn coverage to this same dictionary. A new run from
        run_tournament's policies route records "red_zone_depth" here (the Red
        Zone depth in map units; 0.0 means the rule is off); such a run saved
        before the rule has no such key and played at 0.0. Configured and
        canonical results may omit this convenience key, so a missing key says
        nothing about their depth. New list and short-config results preserve it
        while using the same CanonicalTournamentResult as full configs. For each result,
        game's depth is team_deathmatch_red_zone_depth in its configuration
        under "configurations"; content saved before the rule lacks that field
        and played at 0.0.
    paths : dict of str to Path or None
        Produced files by name, or None for in-memory results.
    headline_metrics : tuple of dict, default=()
        Optional keyword-only headline table, computed by the runner once.
    """

    matches: tuple[Row, ...]
    tournament_results: tuple[Row, ...]
    matchup_results: tuple[Row, ...]
    map_results: tuple[Row, ...]
    full_metrics: Columns
    replays: tuple[ReplayArtifactV4, ...]
    metadata: Row
    paths: dict[str, Path] | None
    headline_metrics: tuple[Row, ...] = field(default=(), kw_only=True)
    _view: _View = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Bind complete tournament rows to their actual pass and system owners."""
        if self.paths and "run_details" in self.paths:
            directory = Path(self.paths["run_details"]).parent
            view = _result_view(
                _read_manifest(directory), run_dir=directory, phase="tournament"
            )
        else:
            manifest = _memory_manifest(self.metadata, tournament=True, completed=())
            memory: Row = {
                "match_results": self.matches,
                "tournament_results": self.tournament_results,
                "matchup_results": self.matchup_results,
                "map_results": self.map_results,
            }
            if "completion_order" in self.metadata:
                memory["completion_order"] = self.metadata["completion_order"]
            if self.full_metrics:
                memory["full_metrics"] = self.full_metrics
            if self.headline_metrics:
                memory["tournament_headline_metrics"] = self.headline_metrics
            if isinstance(self, CanonicalTournamentResult):
                memory["_record_access"] = self._record_access  # pyright: ignore[reportPrivateUsage]
            view = _result_view(
                manifest, run_dir=None, phase="tournament", memory=memory
            )
        object.__setattr__(self, "_view", view)
        self.metadata.update(view.metadata)


@dataclass(frozen=True)
class CanonicalTournamentResult(TournamentResult):
    """Expose a complete selected snapshot through ordinary tournament tables.

    Attributes
    ----------
    snapshot_id : str
        Immutable selected configuration identity; also ``big_12_id``
        in metadata. Fixture snapshots do not establish official qualification.
    challenger_id : str or None
        Logical entrant ID of the optional canonical challenger. None means
        the released field alone for a canonical call. Version 1 official fields
        have twelve entrants; version 2 follows the frozen participant list.
        Custom configuration results may have any supported size. The generic config
        route also supports a separate challenger.
    planned_games, reused_games, executed_games : int
        Whole logical-run counts, including durable games from earlier attempts.
        Metadata ``executed_this_call`` counts only new work in this call.
    games_per_opponent : int
        Uniform resolved number of completed games per unordered participant pair.
    protocol_compliant : bool
        Whether the resolved scientific conditions match the snapshot protocol.
        This is not admission approval or proof of scientific eligibility.

    Notes
    -----
    Original row and replay identities remain unchanged. Uniform tables cover
    reused and newly executed games; inherited full_metrics/replays retain only
    new in-memory captures. No-file results borrow immutable source files for
    lazy access. Keep those files available for the lifetime of the result.
    """

    snapshot_id: str = field(kw_only=True)
    challenger_id: str | None = field(kw_only=True)
    planned_games: int = field(kw_only=True)
    reused_games: int = field(kw_only=True)
    executed_games: int = field(kw_only=True)
    games_per_opponent: int = field(kw_only=True)
    protocol_compliant: bool = field(kw_only=True)
    _record_access: TournamentRecords | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )


@dataclass(frozen=True)
class SavedResults(_ResultAccess):
    """Hold a read-only saved scope with metadata and bounded table methods.

    Instances come from ``load_results``. They own no writer or open file and do
    not promise live updates. A destructive restore invalidates active iteration.
    """

    metadata: Row
    _view: _View = field(repr=False, compare=False)


def load_results(
    run_dir: str | Path, *, phase: str | None = None, pass_id: str | None = None
) -> SavedResults:
    """Read one saved run or an explicitly selected pass without changing it.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Exact directory containing run_details.json. Parent output directories
        are not searched and no newest child is selected.
    phase : str or None, optional
        Select all passes in this phase, default None for the whole run.
    pass_id : str or None, optional
        Select a named pass, default None. Without phase it must be unique.

    Returns
    -------
    SavedResults
        Read-only status, metadata, replay paths and NumPy table access. Missing
        historical facts stay unavailable; no current-schema values are rebuilt.
        Each table's column_types maps names to immutable (type, null-rule)
        tuples; repeated descriptions share storage.

    Raises
    ------
    ValueError
        Path, schema, scope or committed content is invalid, or coordinated
        recording restore is pending.
    OSError
        Files cannot be read. No files, locks, downloads or JAX work are created.
    """
    directory = Path(run_dir)
    view = _result_view(
        _read_manifest(directory), run_dir=directory, phase=phase, pass_id=pass_id
    )
    return SavedResults(view.metadata, view)
