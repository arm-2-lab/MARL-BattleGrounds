"""Check system_metrics: every System's metrics read from its own side.

Contracts: a System that played Team B reads every column from its mirrored
twin (an index-valued game checks all 11,192 columns and the identity
columns), so score_difference and kills equal head_to_head margins and hand
totals; by="opponent" rows match head_to_head rows, and a Team B row's
score_difference and interval are Team A's negated and swapped; recorded Team A
rows give point_margin_summary's interval; self-play counts once; systems=,
columns= and ratios= select and add columns (any non-string iterable, such
as a column of an earlier result) and reject bad names and empty lists; fractions
combine as total over total, and ratios= does the same for researcher-chosen
columns; intervals resample verified pairs (available with finite bounds, at
least one checked), a row whose matchups are all deterministic is withheld
with "No random sampling variation" while a mixed row counts only its
stochastic units, a row with a game of unknown sampling keeps its interval
labeled "Conditional: Sampling not verified" with its stated assumption (and
point_margin_summary gives the same label and bounds), one pair whose passes
have different sampling facts gets one verdict in both summaries (unknown
wins; "No random sampling variation" only when no game is stochastic), and a
rare column gets NaN bounds under an available row;
games played without rows in the table make summary values NaN with a reason
while complete by="opponent" rows stay valid; unplayed games keep descriptive
values but no interval; a slot whose class changes gives NaN slot columns; a
canonical run that reuses another run's games keeps them in pairs for both
system_metrics and point_margin_summary, and saved and in-memory canonical
results agree; pair keys carry the full game origin; the projected CSV read
equals the full read for the kept cells; a disabled table gives {}; and on a
canonical fixture tournament the summaries equal tournament_headline_metrics
(kills, deaths, margin, return, game length), by="opponent" rows come in
head_to_head's order and System rows are sorted by ID.
Synthetic saved runs test software, not learned behavior; no game is played.
"""

# pyright: reportPrivateUsage=false
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.test_result_access import _save, _table

import marl_battlegrounds as marl_bgs
from canonical_record_fixtures import build_record_bundle
from marl_battlegrounds.evaluation.canonical_results import CanonicalView
from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    METRIC_SCHEMA_VERSION,
    mirror_metric,
)
from marl_battlegrounds.evaluation.results import (
    SavedResults,
    _plan_record,
    load_results,
)
from marl_battlegrounds.evaluation.run_writer import (
    EPISODE_COLUMNS,
    IDENTITY_COLUMNS,
    RunWriter,
)
from marl_battlegrounds.evaluation.scalar_reports import iter_scalar_rows
from marl_battlegrounds.evaluation.tournament_assets import (
    AssetVerifier,
    prepare_tournament_assets,
)
from marl_battlegrounds.evaluation.tournament_records import TournamentRecords
from marl_battlegrounds.evaluation.tournament_reuse import resolve_reuse_plan

FULL_HEADER = (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES)
NAMES = {"s1": "One", "s2": "Two", "s3": "Three"}
type Values = Callable[[int, int, int, bool], Mapping[str, float | None]]


def _values(matchup: int, spawn: int, pair: int, fixed: bool) -> dict[str, float]:
    step = matchup * 7 + spawn * 3 + (0 if fixed else pair * 5)
    a, b = 10 + step % 6, 8 + (step * 3) % 5
    return {
        "team_a_score": a,
        "team_b_score": b,
        "score_difference": a - b,
        "team_a_kills": a,
        "team_b_kills": b,
        "team_a_deaths": b,
        "team_b_deaths": a,
        "team_a_multi_contributor_kills": a // 2,
        "team_b_multi_contributor_kills": b // 3,
        "team_a_multi_contributor_kill_fraction": (a // 2) / a,
        "team_b_multi_contributor_kill_fraction": (b // 3) / b,
        "agent_0_damage_done": 100 + step,
        "agent_5_damage_done": 50 + step,
    }


def _run(
    tmp_path: Path,
    matchups: Sequence[tuple[str, str, str]],
    *,
    pairs: int = 3,
    maps: Sequence[int] = (47, 48),
    values: Values = _values,
    full: Callable[[int, int], bool] = lambda matchup, episode: True,
    unplayed: Sequence[tuple[int, int]] = (),
    classes: Callable[[int, int], Sequence[int]] = lambda matchup, episode: (
        (
            1,
            2,
            3,
            4,
            5,
        )
        * 2
    ),
    names: Mapping[str, str] = NAMES,
    metrics: str = "full",
) -> Path:
    directory = tmp_path / "run"
    directory.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "metric_schema_id": "marlbg.tdm.scalar",
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "run_id": "run",
        "systems": {},
        "configurations": {"cfg": {}},
        "tables": {},
        "details": {},
        "passes": {},
    }
    outcomes: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for matchup, (a, b, sampling) in enumerate(matchups):
        entry: dict[str, Any] = {
            "phase": "evaluation",
            "pass_id": f"m{matchup}",
            "policies": {"team_a": {"name": names[a]}, "team_b": {"name": names[b]}},
            "system_ids": {"team_a": a, "team_b": b},
            "details": {
                "metrics": metrics,
                "seed": 100,
                "rng_protocol": "evaluation-systems-v1",
                "method_sampling": {
                    team: {"determinism": sampling, "basis": "Software fixture"}
                    for team in ("team_a", "team_b")
                },
                "evaluation_contract": {
                    "version": 1,
                    "spawn_mode": "paired",
                    "pairing_protocol": "fixed-team-spawn-v1",
                },
            },
            "episodes": {},
            "completed_episode_ids": [],
            "replays": {},
            "recorded_metrics_by_episode": {},
            "result_state": {
                "version": 1,
                "status": "complete",
                "schedule_digest": "digest",
                "reason": None,
            },
        }
        episode = 0
        for map_id in maps:
            for pair in range(pairs):
                for spawn in (0, 1):
                    episode += 1
                    seed_id = matchup * 1000 + map_id * 10 + pair
                    entry["episodes"][str(episode)] = {
                        "seed_id": seed_id,
                        "map_id": map_id,
                        "config_id": "cfg",
                        "spawn_locations": spawn,
                        "comparison_kind": "verified_spawn_pair",
                        "paired_comparison_key": f"pair-{map_id}-{pair}",
                        "bootstrap_group": None,
                    }
                    if (matchup, episode) in unplayed:
                        entry["result_state"]["status"] = "incomplete"
                        continue
                    measured = values(matchup, spawn, pair, sampling != "stochastic")
                    identity: dict[str, Any] = {
                        "run_id": "run",
                        "phase": "evaluation",
                        "pass_id": f"m{matchup}",
                        "episode_id": episode,
                        "seed_id": seed_id,
                        "map_id": map_id,
                        "config_id": "cfg",
                        "team_a_policy": names[a],
                        "team_b_policy": names[b],
                        "checkpoint_id": None,
                    }
                    for slot, class_id in enumerate(classes(matchup, episode)):
                        identity[f"agent_{slot}_class_id"] = class_id
                        identity[f"agent_{slot}_active"] = 1
                    outcomes.append(
                        {
                            **identity,
                            "outcome": 1,
                            "episode_length": 10,
                            "team_a_score": measured["team_a_score"],
                            "team_b_score": measured["team_b_score"],
                        }
                    )
                    entry["completed_episode_ids"].append(episode)
                    covered = full(matchup, episode)
                    entry["recorded_metrics_by_episode"][str(episode)] = (
                        "full" if covered else "none"
                    )
                    if covered:
                        reports.append({**identity, **measured})
        manifest["passes"][str(matchup)] = entry
    _table(directory, manifest, "episodes.csv", EPISODE_COLUMNS, outcomes)
    _table(directory, manifest, "full_metrics.csv", FULL_HEADER, reports)
    _save(directory, manifest)
    return directory


def _rows(
    columns: Mapping[str, np.ndarray], *keys: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    size = len(next(iter(columns.values())))
    return {
        tuple(columns[key][index] for key in keys): {
            name: values[index] for name, values in columns.items()
        }
        for index in range(size)
    }


TRIANGLE = [
    ("s1", "s2", "stochastic"),
    ("s1", "s3", "stochastic"),
    ("s3", "s2", "stochastic"),
]


def test_every_column_reads_from_the_systems_side(tmp_path: Path) -> None:
    def indexed(matchup: int, spawn: int, pair: int, fixed: bool) -> dict[str, float]:
        return {
            **{name: float(place + 1) for place, name in enumerate(FULL_METRIC_NAMES)},
            "team_a_score": 3,
            "team_b_score": 1,
        }

    path = _run(
        tmp_path,
        [("s1", "s2", "stochastic")],
        pairs=1,
        maps=(47,),
        values=indexed,
        classes=lambda matchup, episode: (1, 2, 3, 4, 5, 5, 4, 3, 2, 1),
    )
    games = load_results(path).system_metrics("full_metrics", by="game")
    position = {name: place + 1 for place, name in enumerate(FULL_METRIC_NAMES)}
    assert games["system_id"].tolist() == ["s1", "s2", "s1", "s2"]
    assert games["played_as"].tolist() == ["team_a", "team_b"] * 2
    assert games["spawn_end"].tolist() == ["team_a", "team_b", "team_b", "team_a"]
    assert games["team_a_policy"].tolist() == ["One", "Two", "One", "Two"]
    assert games["agent_0_class_id"].tolist() == [1, 5, 1, 5]
    assert games["team_a_score"].tolist() == [3, 1, 3, 1]
    for name in FULL_METRIC_NAMES:
        if name in ("team_a_score", "team_b_score"):
            continue
        twin, sign = mirror_metric(name)
        assert games[name][0] == position[name], name
        assert games[name][1] == sign * position[twin], name


def test_margins_kills_and_rows_match_head_to_head(tmp_path: Path) -> None:
    results = load_results(_run(tmp_path, TRIANGLE))
    head = results.head_to_head()
    by_pair = _rows(head, "system_id", "opponent_id")
    opponents = _rows(
        results.system_metrics("full_metrics", by="opponent"),
        "system_id",
        "opponent_id",
    )
    assert set(opponents) == set(by_pair)
    for key, row in by_pair.items():
        assert opponents[key]["games"] == row["games"]
        assert opponents[key]["system_name"] == row["system_name"]
        assert opponents[key]["score_difference"] == pytest.approx(
            row["mean_point_margin"]
        )
    systems = _rows(results.system_metrics("full_metrics"), "system_id")
    for system in NAMES:
        rows = [row for key, row in by_pair.items() if key[0] == system]
        games = sum(row["games"] for row in rows)
        margin = sum(row["games"] * row["mean_point_margin"] for row in rows) / games
        points = sum(row["games"] * row["mean_points_for"] for row in rows) / games
        assert systems[(system,)]["games"] == games
        assert systems[(system,)]["score_difference"] == pytest.approx(margin)
        # In this fixture kills equal points, from each System's own side.
        assert systems[(system,)]["team_a_kills"] == pytest.approx(points)


def test_team_b_rows_and_point_margin_intervals_agree(tmp_path: Path) -> None:
    results = load_results(_run(tmp_path, TRIANGLE))
    opponents = _rows(
        results.system_metrics(
            "full_metrics", by="opponent", columns=["score_difference"]
        ),
        "system_id",
        "opponent_id",
    )
    margins = _rows(results.point_margin_summary(), "system_id", "opponent_id")
    for (a, b), row in margins.items():
        forward, backward = opponents[(a, b)], opponents[(b, a)]
        assert forward["interval_status"] == "Available"
        assert forward["score_difference_ci_low"] == pytest.approx(
            row["ci_low"], abs=1e-9
        )
        assert forward["score_difference_ci_high"] == pytest.approx(
            row["ci_high"], abs=1e-9
        )
        assert backward["score_difference"] == pytest.approx(
            -forward["score_difference"]
        )
        assert backward["score_difference_ci_low"] == pytest.approx(
            -forward["score_difference_ci_high"], abs=1e-9
        )
        assert backward["score_difference_ci_high"] == pytest.approx(
            -forward["score_difference_ci_low"], abs=1e-9
        )
        assert forward["independent_units"] == backward["independent_units"] == 6


def test_self_play_counts_once(tmp_path: Path) -> None:
    results = load_results(_run(tmp_path, [("s1", "s1", "stochastic")], pairs=2))
    games = results.system_metrics("full_metrics", by="game", columns=["team_a_kills"])
    assert games["system_id"].tolist() == ["s1"] * 8
    assert games["played_as"].tolist() == ["team_a"] * 8
    summary = results.system_metrics("full_metrics", columns=["team_a_kills"])
    assert summary["games"].tolist() == [8]
    assert summary["scheduled_games"].tolist() == [8]


def test_selection_ratios_and_errors(tmp_path: Path) -> None:
    results = load_results(_run(tmp_path, TRIANGLE))
    picked = results.system_metrics(
        "full_metrics",
        systems=["Two"],
        columns=["team_a_kills"],
        ratios={"shared": ("team_a_multi_contributor_kills", "team_a_kills")},
    )
    assert picked["system_id"].tolist() == ["s2"]
    # A column of an earlier result works as a selection.
    chained = results.system_metrics(
        "full_metrics",
        systems=picked["system_id"],
        columns=np.asarray(["team_a_kills"]),
    )
    assert chained["system_id"].tolist() == ["s2"]
    assert [name for name in picked if name.startswith(("team_a_", "shared"))] == [
        "team_a_kills",
        "team_a_kills_ci_low",
        "team_a_kills_ci_high",
        "shared",
        "shared_ci_low",
        "shared_ci_high",
    ]
    assert picked["shared"][0] == pytest.approx(
        picked_fraction := results.system_metrics(
            "full_metrics",
            systems=["s2"],
            columns=["team_a_multi_contributor_kill_fraction"],
        )["team_a_multi_contributor_kill_fraction"][0],
        rel=1e-6,
    )
    games = results.system_metrics("full_metrics", by="game", systems=["s2"])
    shared = games["team_a_multi_contributor_kills"].astype(np.float64)
    kills = games["team_a_kills"].astype(np.float64)
    assert picked_fraction == pytest.approx(shared.sum() / kills.sum(), rel=1e-6)
    assert picked_fraction != pytest.approx(np.mean(shared / kills), rel=1e-6)
    cases: list[tuple[dict[str, Any], str]] = [
        ({"systems": ["nobody"]}, "No System here"),
        ({"systems": "s1"}, "list of nonempty strings"),
        ({"columns": ["team_a_policy"]}, "not metric columns"),
        ({"columns": ["team_a_kills", "team_a_kills"]}, "repeat"),
        ({"columns": []}, "nonempty list"),
        (
            {"ratios": {"team_a_kills": ("team_a_kills", "team_a_deaths")}},
            "already a column",
        ),
        (
            {"ratios": {"x_ci_low": ("team_a_kills", "team_a_deaths")}},
            "already a column",
        ),
        ({"ratios": {"x": ("team_a_kills", "nope")}}, "not in the full_metrics table"),
        ({"ratios": {"x": "team_a_kills"}}, "numerator, denominator"),
        ({"by": "overall"}, "by must be"),
        ({"bootstrap_draws": 1}, "bootstrap_draws"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            results.system_metrics("full_metrics", **kwargs)
    with pytest.raises(ValueError, match="table must be"):
        results.system_metrics("matches")


def test_shared_display_name_is_refused(tmp_path: Path) -> None:
    results = load_results(
        _run(tmp_path, TRIANGLE, names={"s1": "One", "s2": "Same", "s3": "Same"})
    )
    with pytest.raises(ValueError, match="names several Systems"):
        results.system_metrics("full_metrics", systems=["Same"])


def test_intervals_follow_sampling_facts(tmp_path: Path) -> None:
    mixed = [("s1", "s2", "deterministic"), ("s1", "s3", "stochastic")]
    results = load_results(_run(tmp_path, mixed))
    systems = _rows(
        results.system_metrics("full_metrics", columns=["score_difference"]),
        "system_id",
    )
    one = systems[("s1",)]
    assert one["interval_status"] == "Available"
    assert np.isfinite(one["score_difference_ci_low"])
    assert one["score_difference_ci_low"] < one["score_difference_ci_high"]
    assert one["independent_units"] == 6
    two = systems[("s2",)]
    assert two["interval_status"] == "Unavailable"
    assert two["unavailable_reason"] == "No random sampling variation"
    assert two["independent_units"] == 0
    assert np.isfinite(two["score_difference"])
    opponents = _rows(
        results.system_metrics(
            "full_metrics", by="opponent", columns=["score_difference"]
        ),
        "system_id",
        "opponent_id",
    )
    assert (
        opponents[("s1", "s2")]["unavailable_reason"] == "No random sampling variation"
    )


@pytest.mark.parametrize(
    ("samplings", "status", "reason"),
    [
        (("unknown", "deterministic"), "Conditional: Sampling not verified", None),
        (("stochastic", "deterministic"), "Available", None),
        (
            ("deterministic", "deterministic"),
            "Unavailable",
            "No random sampling variation",
        ),
    ],
)
def test_mixed_passes_give_both_summaries_one_sampling_verdict(
    tmp_path: Path, samplings: tuple[str, str], status: str, reason: str | None
) -> None:
    results = load_results(_run(tmp_path, [("s1", "s2", kind) for kind in samplings]))
    margin = _rows(results.point_margin_summary(), "system_id", "opponent_id")[
        ("s1", "s2")
    ]
    metric = _rows(
        results.system_metrics(
            "full_metrics", by="opponent", columns=["score_difference"]
        ),
        "system_id",
        "opponent_id",
    )[("s1", "s2")]
    assert margin["interval_status"] == metric["interval_status"] == status
    assert margin["unavailable_reason"] == metric["unavailable_reason"] == reason
    if reason is None:
        assert margin["ci_low"] == pytest.approx(
            metric["score_difference_ci_low"], abs=1e-9
        )
        assert margin["ci_high"] == pytest.approx(
            metric["score_difference_ci_high"], abs=1e-9
        )
    else:
        assert np.isnan(margin["ci_low"]) and np.isnan(
            metric["score_difference_ci_low"]
        )


def test_unknown_sampling_keeps_intervals_labeled_conditional(tmp_path: Path) -> None:
    mixed = [("s1", "s2", "unknown"), ("s1", "s3", "stochastic")]
    results = load_results(_run(tmp_path, mixed))
    systems = _rows(
        results.system_metrics("full_metrics", columns=["score_difference"]),
        "system_id",
    )
    for name in ("s1", "s2"):
        row = systems[(name,)]
        assert row["interval_status"] == "Conditional: Sampling not verified"
        assert row["unavailable_reason"] is None
        assert row["conditional_interval_assumption"] == (
            "Assumes declared game groups are independent; sampling is not verified"
        )
        assert row["independent_units"] is None
        low, high = row["score_difference_ci_low"], row["score_difference_ci_high"]
        assert np.isfinite(low) and low <= row["score_difference"] <= high
    # s2's fixture games repeat exactly; s1 also has random games against s3.
    assert (
        systems[("s1",)]["score_difference_ci_low"]
        < (systems[("s1",)]["score_difference_ci_high"])
    )
    three = systems[("s3",)]
    assert three["interval_status"] == "Available"
    assert three["conditional_interval_assumption"] is None
    opponents = _rows(
        results.system_metrics(
            "full_metrics", by="opponent", columns=["score_difference"]
        ),
        "system_id",
        "opponent_id",
    )
    margins = _rows(results.point_margin_summary(), "system_id", "opponent_id")
    for key, status in ((("s1", "s2"), "Conditional"), (("s1", "s3"), "Available")):
        forward, margin = opponents[key], margins[key]
        assert forward["interval_status"].startswith(status)
        assert margin["interval_status"] == forward["interval_status"]
        assert (
            margin["conditional_interval_assumption"]
            == (forward["conditional_interval_assumption"])
        )
        assert forward["score_difference_ci_low"] == pytest.approx(
            margin["ci_low"], abs=1e-9
        )
        assert forward["score_difference_ci_high"] == pytest.approx(
            margin["ci_high"], abs=1e-9
        )


def test_rare_column_has_nan_bounds_under_an_available_row(tmp_path: Path) -> None:
    def rare(
        matchup: int, spawn: int, pair: int, fixed: bool
    ) -> dict[str, float | None]:
        measured: dict[str, float | None] = {**_values(matchup, spawn, pair, fixed)}
        chances = 2.0 if (pair, spawn) == (0, 0) else 0.0
        measured["team_a_rescue_opportunities"] = chances
        measured["team_a_rescue_rate"] = 0.5 if chances else None
        return measured

    results = load_results(_run(tmp_path, [("s1", "s2", "stochastic")], values=rare))
    row = _rows(
        results.system_metrics("full_metrics", columns=["team_a_rescue_rate"]),
        "system_id",
    )[("s1",)]
    assert row["interval_status"] == "Available"
    assert row["team_a_rescue_rate"] == pytest.approx(0.5)
    assert np.isnan(row["team_a_rescue_rate_ci_low"])
    assert np.isnan(row["team_a_rescue_rate_ci_high"])


def test_partial_coverage_and_unplayed_games(tmp_path: Path) -> None:
    partial = load_results(
        _run(tmp_path / "partial", TRIANGLE, full=lambda matchup, episode: matchup != 1)
    )
    systems = _rows(
        partial.system_metrics("full_metrics", columns=["team_a_kills"]), "system_id"
    )
    one = systems[("s1",)]
    assert np.isnan(one["team_a_kills"])
    assert one["games"] == 12
    assert "Full metrics cover only 12 of 24 played games" in one["unavailable_reason"]
    opponents = _rows(
        partial.system_metrics("full_metrics", by="opponent", columns=["team_a_kills"]),
        "system_id",
        "opponent_id",
    )
    assert np.isfinite(opponents[("s1", "s2")]["team_a_kills"])
    assert opponents[("s1", "s2")]["interval_status"] == "Available"
    assert np.isnan(opponents[("s1", "s3")]["team_a_kills"])
    unplayed = load_results(
        _run(tmp_path / "unplayed", [("s1", "s2", "stochastic")], unplayed=[(0, 3)])
    )
    row = _rows(
        unplayed.system_metrics("full_metrics", columns=["team_a_kills"]), "system_id"
    )[("s1",)]
    assert row["games"] == 11
    assert row["scheduled_games"] == 12
    assert np.isfinite(row["team_a_kills"])
    assert "Requested games are incomplete" in row["unavailable_reason"]


def test_changed_slot_class_blanks_slot_columns(tmp_path: Path) -> None:
    def classes(matchup: int, episode: int) -> Sequence[int]:
        first = 1 if episode < 3 else 2
        return (first, 2, 3, 4, 5, 1, 2, 3, 4, 5)

    results = load_results(
        _run(tmp_path, [("s1", "s2", "stochastic")], classes=classes)
    )
    columns = ["agent_0_damage_done", "agent_5_damage_done", "team_a_kills"]
    rows = _rows(results.system_metrics("full_metrics", columns=columns), "system_id")
    assert rows[("s1",)]["agent_0_class_id"] is None
    assert np.isnan(rows[("s1",)]["agent_0_damage_done"])
    assert np.isfinite(rows[("s1",)]["agent_5_damage_done"])
    assert np.isfinite(rows[("s1",)]["team_a_kills"])
    assert np.isfinite(rows[("s2",)]["agent_0_damage_done"])
    assert rows[("s2",)]["agent_5_class_id"] is None
    games = results.system_metrics("full_metrics", by="game", systems=["s1"])
    assert np.all(np.isfinite(games["agent_0_damage_done"]))


def _canonical(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], TournamentRecords]:
    bundle = build_record_bundle(
        tmp_path / "source", entrants=3, maps=1, full=True, pairs_per_map=2
    )
    config = bundle["config"]
    plan = resolve_reuse_plan(config, bundle["paths"])
    with RunWriter(
        output_dir=tmp_path / "runs",
        phase="tournament",
        pass_id="schedule",
        details={
            "metrics": "full",
            "full_metrics_episodes": [],
            "replay_episodes": [],
            "snapshot_id": config["snapshot_id"],
        },
    ) as writer:
        writer.set_tournament_coordinator("schedule")
        writer._install_tournament_plan(
            config,
            plan.games,
            {
                "version": 1,
                "source_descriptors": list(plan.source_descriptors),
                "execution_plan": list(plan.jobs),
                "budget": plan.budget,
                "challenger_id": None,
                "state": "incomplete",
            },
        )
        path = writer.run_dir
    manifest = json.loads((path / "run_details.json").read_bytes())
    records = TournamentRecords(
        config,
        plan.games,
        plan.jobs,
        AssetVerifier(config),
        manifest=manifest,
        run_dir=path,
    )
    return path, manifest, records


def test_canonical_reuse_keeps_reused_games_in_pairs(tmp_path: Path) -> None:
    path, manifest, records = _canonical(tmp_path)
    saved = load_results(path)
    columns = ["score_difference", "team_a_kills"]
    summary = saved.system_metrics("full_metrics", columns=columns)
    assert sorted(summary["system_id"].tolist()) == [
        "fixture-00",
        "fixture-01",
        "fixture-02",
    ]
    assert summary["scheduled_games"].tolist() == [8, 8, 8]
    assert summary["completed_pairs"].tolist() == [4, 4, 4]
    assert summary["interval_status"].tolist() == ["Available"] * 3
    margins = saved.point_margin_summary()
    assert margins["completed_pairs"].tolist() == [2, 2, 2]
    assert margins["interval_status"].tolist() == ["Available"] * 3
    view = CanonicalView(manifest, run_dir=None, memory={"_record_access": records})
    memory = SavedResults(view.metadata, view).system_metrics(
        "full_metrics", columns=columns
    )
    for name, values in summary.items():
        np.testing.assert_equal(values, memory[name])


def test_pair_keys_carry_the_full_origin() -> None:
    facts = {
        "pass_complete": True,
        "root_seed": 1,
        "root_known": True,
        "sampling": "stochastic",
    }
    declared = {
        "comparison_kind": "verified_spawn_pair",
        "paired_comparison_key": "pair-1",
        "seed_id": 1,
        "bootstrap_group": None,
    }
    context = ("a", "b", "A", "B", 0)
    local = _plan_record(
        ("local", "tournament", "canonical-job-0", "1"), facts, declared, None, context
    )
    reused = _plan_record(
        ("source", "tournament", "canonical-job-0", "1"), facts, declared, None, context
    )
    assert local["pair"] != reused["pair"]
    assert local["unit"] != reused["unit"]
    blocked = {**declared, "bootstrap_group": "block-7"}
    assert (
        _plan_record(("local", "t", "p", "1"), facts, blocked, None, context)["unit"]
        == _plan_record(("source", "t", "q", "1"), facts, blocked, None, context)[
            "unit"
        ]
    )


def test_projected_read_equals_full_read(tmp_path: Path) -> None:
    path = _run(tmp_path, [("s1", "s2", "stochastic")], pairs=1, maps=(47,))
    manifest = json.loads((path / "run_details.json").read_bytes())
    keep = {*IDENTITY_COLUMNS, "team_a_kills", "agent_5_damage_done"}
    full = [
        row
        for batch in iter_scalar_rows(
            path / "full_metrics.csv", manifest=manifest, expected_header=FULL_HEADER
        )
        for row in batch
    ]
    narrow = [
        row
        for batch in iter_scalar_rows(
            path / "full_metrics.csv",
            manifest=manifest,
            expected_header=FULL_HEADER,
            columns=keep,
        )
        for row in batch
    ]
    assert narrow == [{name: row[name] for name in row if name in keep} for row in full]


def test_disabled_table_gives_an_empty_result(tmp_path: Path) -> None:
    results = load_results(
        _run(
            tmp_path,
            [("s1", "s2", "stochastic")],
            full=lambda m, e: False,
            metrics="none",
        )
    )
    assert results.system_metrics("full_metrics") == {}


def test_summaries_match_headlines_and_head_to_head_order(tmp_path: Path) -> None:
    bundle = build_record_bundle(
        tmp_path / "source", entrants=4, maps=1, pairs_per_map=2
    )
    prepared = prepare_tournament_assets(
        bundle["config"], roles=("outcomes_priority",), download=False
    )
    result = marl_bgs.run_tournament(config=prepared["config"], metrics="priority")
    headlines = _rows(result.table("tournament_headline_metrics"), "system_id")
    systems = _rows(result.system_metrics("priority_metrics"), "system_id")
    assert set(systems) == set(headlines)
    for key, row in headlines.items():
        for mean, column in (
            ("kills_mean", "team_a_kills"),
            ("deaths_mean", "team_a_deaths"),
            ("score_difference_mean", "score_difference"),
            ("mean_policy_return", "team_a_return"),
            ("episode_length_mean", "episode_length"),
        ):
            assert systems[key][column] == pytest.approx(row[mean]), column
    head = result.head_to_head()
    rows = result.system_metrics(
        "priority_metrics", by="opponent", columns=["score_difference"]
    )
    assert rows["system_id"].tolist() == head["system_id"].tolist()
    assert rows["opponent_id"].tolist() == head["opponent_id"].tolist()
    ordered = result.system_metrics("priority_metrics")["system_id"].tolist()
    assert ordered == sorted(key[0] for key in systems)
