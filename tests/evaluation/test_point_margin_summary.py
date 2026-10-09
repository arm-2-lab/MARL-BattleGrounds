"""Check native-point intervals from saved pairs without running policies.

Game weights, Team A ownership, larger linked blocks and incomplete evidence
remain visible. Unknown sampling keeps the interval, labeled "Conditional:
Sampling not verified" with its stated assumption; deterministic games withhold
it, and so does unknown sampling beside another problem (every reason joined,
no assumption). Only the requested accessor resamples; cheap descriptions do not. These
synthetic fixtures test software, not learned behavior.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# pyright: reportPrivateUsage=false
from tests.evaluation.test_result_access import _fixture, _row, _save, _table

from marl_battlegrounds.evaluation.results import load_results
from marl_battlegrounds.evaluation.run_writer import EPISODE_COLUMNS


def _saved(
    tmp_path: Path,
    pairs: Sequence[tuple[int, float, str | None]],
    *,
    omit: int | None = None,
    sampling: str = "stochastic",
) -> Path:
    directory, manifest = _fixture(tmp_path, modes=("none",))
    entry = manifest["passes"]["0"]
    entry["details"].update(seed=123, rng_protocol="evaluation-systems-v1")
    entry["details"]["method_sampling"] = {
        name: {"determinism": sampling, "basis": "Declared software fixture"}
        for name in ("team_a", "team_b")
    }
    entry["episodes"] = {}
    rows: list[dict[str, Any]] = []
    for index, (map_id, margin, block) in enumerate(pairs):
        for spawn in (0, 1):
            identifier = index * 2 + spawn + 1
            entry["episodes"][str(identifier)] = {
                "seed_id": index,
                "map_id": map_id,
                "config_id": "cfg",
                "spawn_locations": spawn,
                "comparison_kind": "verified_spawn_pair",
                "paired_comparison_key": f"pair-{index}",
                "bootstrap_group": block,
            }
            if identifier != omit:
                row = _row(episode_id=identifier)
                row.update(
                    map_id=map_id,
                    seed_id=index,
                    team_a_score=10 + margin,
                    team_b_score=10,
                )
                rows.append(row)
    entry["completed_episode_ids"] = [row["episode_id"] for row in rows]
    if omit is not None:
        entry["result_state"]["status"] = "incomplete"
    _table(directory, manifest, "episodes.csv", EPISODE_COLUMNS, rows)
    _save(directory, manifest)
    return directory


def test_unequal_map_budgets_weight_games_and_keep_team_a(tmp_path: Path) -> None:
    directory = _saved(tmp_path, [(0, 0, None)] * 2 + [(1, 6, None)] * 4)
    before = {p: p.read_bytes() for p in directory.iterdir()}
    result = load_results(directory)
    summary = result.point_margin_summary(bootstrap_draws=100)
    assert summary["system_id"].tolist() == ["a"]
    assert summary["games"].tolist() == [12]
    assert summary["completed_pairs"].tolist() == [6]
    assert summary["mean_point_margin"].tolist() == [4]
    assert summary["ci_low"].tolist() == summary["ci_high"].tolist() == [4]
    assert result.point_margin_summary(by="map")["mean_point_margin"].tolist() == [0, 6]
    assert all(path.read_bytes() == data for path, data in before.items())


def test_linked_blocks_move_together_across_maps(tmp_path: Path) -> None:
    directory = _saved(
        tmp_path, [(0, 4, "one"), (1, -4, "one"), (0, -4, "two"), (1, 4, "two")]
    )
    result = load_results(directory)
    first = result.point_margin_summary(bootstrap_draws=100, bootstrap_seed=99)
    second = result.point_margin_summary(bootstrap_draws=100, bootstrap_seed=99)
    assert first["mean_point_margin"].tolist() == [0]
    assert first["ci_low"].tolist() == first["ci_high"].tolist() == [0]
    assert first["independent_units"].tolist() == [2]
    for name in first:
        np.testing.assert_equal(first[name], second[name])
    maps = result.point_margin_summary(by="map", bootstrap_draws=100)
    assert maps["ci_low"].tolist() == [-4, -4]
    assert maps["ci_high"].tolist() == [4, 4]


@pytest.mark.parametrize("sampling", ["unknown", "deterministic"])
def test_sampling_facts_label_or_withhold_intervals_not_observed_means(
    tmp_path: Path, sampling: str
) -> None:
    result = load_results(
        _saved(tmp_path, [(0, 1, None), (0, 5, None)], sampling=sampling)
    )
    summary = result.point_margin_summary()
    assert summary["mean_point_margin"].tolist() == [3]
    if sampling == "deterministic":
        assert np.isnan(summary["ci_low"]).all()
        assert summary["interval_status"].tolist() == ["Unavailable"]
        assert summary["unavailable_reason"][0] == "No random sampling variation"
        assert summary["conditional_interval_assumption"][0] is None
    else:
        assert summary["ci_low"].tolist() == [1] and summary["ci_high"].tolist() == [5]
        assert summary["interval_status"].tolist() == [
            "Conditional: Sampling not verified"
        ]
        assert summary["unavailable_reason"][0] is None
        assert summary["conditional_interval_assumption"][0].startswith(
            "Assumes declared game groups are independent"
        )


def test_unknown_sampling_beside_another_problem_withholds_interval(
    tmp_path: Path,
) -> None:
    result = load_results(
        _saved(tmp_path, [(0, 1, None), (0, 5, None)], omit=1, sampling="unknown")
    )
    summary = result.point_margin_summary()
    assert summary["interval_status"].tolist() == ["Unavailable"]
    assert np.isnan(summary["ci_low"]).all() and np.isnan(summary["ci_high"]).all()
    reasons = summary["unavailable_reason"][0].split("; ")
    assert reasons[0] == "Requested pass is incomplete or failed"
    assert reasons[1] == "Sampling mechanism unknown"
    assert "Requested games are incomplete" in reasons
    assert summary["conditional_interval_assumption"][0] is None


def test_missing_pair_stays_visible_and_resume_can_complete_it(tmp_path: Path) -> None:
    directory = _saved(tmp_path, [(0, 1, None), (0, 5, None)], omit=4)
    summary = load_results(directory).point_margin_summary()
    assert summary["games"].tolist() == [3]
    assert summary["scheduled_games"].tolist() == [4]
    assert summary["mean_point_margin"].tolist() == [7 / 3]
    assert np.isnan(summary["ci_low"]).all()
    # Publish the missing durable row using the fixture writer, then open a new
    # immutable saved snapshot as real recovery does.
    manifest = json.loads((directory / "run_details.json").read_text())
    rows: list[dict[str, Any]] = []
    for identifier, margin in ((1, 1), (2, 1), (3, 5), (4, 5)):
        row = _row(episode_id=identifier)
        row.update(
            seed_id=(identifier - 1) // 2, team_a_score=10 + margin, team_b_score=10
        )
        rows.append(row)
    manifest["passes"]["0"]["completed_episode_ids"] = [1, 2, 3, 4]
    manifest["passes"]["0"]["result_state"]["status"] = "complete"
    _table(directory, manifest, "episodes.csv", EPISODE_COLUMNS, rows)
    _save(directory, manifest)
    complete = load_results(directory).point_margin_summary()
    assert complete["interval_status"].tolist() == ["Available"]
    assert complete["mean_point_margin"].tolist() == [3]


def test_each_stratum_needs_two_units_and_plain_summary_skips_resampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = load_results(_saved(tmp_path, [(0, 1, None), (0, 2, None), (1, 3, None)]))
    summary = result.point_margin_summary()
    assert np.isnan(summary["ci_low"]).all()
    assert "stratum" in summary["unavailable_reason"][0]

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Cheap summary performed bootstrap work")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    assert result.head_to_head()["games"].tolist() == [6, 6]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"by": "spawn"},
        {"bootstrap_draws": 1},
        {"bootstrap_draws": True},
        {"bootstrap_seed": -1},
        {"bootstrap_seed": True},
        {"bootstrap_seed": 1.5},
        {"bootstrap_seed": 2**63},
    ],
)
def test_invalid_summary_settings_fail(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    result = load_results(_saved(tmp_path, [(0, 1, None), (0, 2, None)]))
    with pytest.raises(ValueError):
        result.point_margin_summary(**kwargs)


def test_repeated_seed_coordinates_need_one_explicit_block(tmp_path: Path) -> None:
    directory = _saved(tmp_path, [(0, 1, None), (0, 5, None)])
    manifest = json.loads((directory / "run_details.json").read_text())
    for row in manifest["passes"]["0"]["episodes"].values():
        row["seed_id"] = 0
    rows = [_row(episode_id=index) for index in range(1, 5)]
    for row in rows:
        row.update(
            seed_id=0, team_a_score=11 if row["episode_id"] < 3 else 15, team_b_score=10
        )
    _table(directory, manifest, "episodes.csv", EPISODE_COLUMNS, rows)
    _save(directory, manifest)
    summary = load_results(directory).point_margin_summary()
    assert np.isnan(summary["ci_low"]).all()
    assert "Repeated seed" in summary["unavailable_reason"][0]


def test_changed_native_conditions_are_separate_sampling_strata(tmp_path: Path) -> None:
    directory = _saved(tmp_path, [(0, 1, None), (0, 5, None)])
    manifest = json.loads((directory / "run_details.json").read_text())
    for identifier, row in manifest["passes"]["0"]["episodes"].items():
        row["source_config_id"] = "depth-one" if int(identifier) < 3 else "depth-two"
    _save(directory, manifest)
    summary = load_results(directory).point_margin_summary(by="map")
    assert summary["mean_point_margin"].tolist() == [3]
    assert np.isnan(summary["ci_low"]).all()
    assert "stratum" in summary["unavailable_reason"][0]


@pytest.mark.parametrize("protocol", ["episode-fold-in-v1", "evaluation-systems-v1"])
def test_same_random_coordinates_in_two_passes_do_not_create_new_units(
    tmp_path: Path,
    protocol: str,
) -> None:
    import copy

    directory = _saved(tmp_path, [(0, 1, None)])
    manifest = json.loads((directory / "run_details.json").read_text())
    repeated = copy.deepcopy(manifest["passes"]["0"])
    repeated["pass_id"] = "repeat"
    repeated["details"]["rng_protocol"] = protocol
    manifest["passes"]["1"] = repeated
    rows: list[dict[str, Any]] = []
    for pass_id in ("same", "repeat"):
        for identifier in (1, 2):
            row = _row(episode_id=identifier)
            row.update(pass_id=pass_id, team_a_score=11, team_b_score=10)
            rows.append(row)
    _table(directory, manifest, "episodes.csv", EPISODE_COLUMNS, rows)
    _save(directory, manifest)
    summary = load_results(directory).point_margin_summary()
    assert summary["games"].tolist() == [4]
    assert np.isnan(summary["ci_low"]).all()
    assert "Repeated seed" in summary["unavailable_reason"][0]


def test_unequal_block_counts_keep_linked_cells_and_original_game_weights(
    tmp_path: Path,
) -> None:
    # Every block covers both maps, with different numbers of games. Map 0
    # always scores +2, map 1 always scores -4; original shares are each one half.
    directory = _saved(
        tmp_path,
        [
            (0, 2, "a"),
            (1, -4, "a"),
            (1, -4, "a"),
            (0, 2, "b"),
            (0, 2, "b"),
            (1, -4, "b"),
            (0, 2, "c"),
            (1, -4, "c"),
        ],
    )
    summary = load_results(directory).point_margin_summary(bootstrap_draws=100)
    assert summary["independent_units"].tolist() == [3]
    assert summary["interval_status"].tolist() == ["Available"]
    assert summary["mean_point_margin"].tolist() == [-1]
    assert summary["ci_low"].tolist() == summary["ci_high"].tolist() == [-1]


def test_completion_is_checked_once_per_pass_not_per_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation import results

    directory = _saved(tmp_path, [(0, 1, None)] * 100)
    manifest = json.loads((directory / "run_details.json").read_text())
    manifest["passes"]["0"].pop("result_state")
    _save(directory, manifest)
    result = load_results(directory)
    original = results._state
    calls: list[int] = []

    def counted(
        entry: dict[str, Any], manifest: Mapping[str, Any]
    ) -> tuple[str, str | None]:
        calls.append(1)
        return original(entry, manifest)

    monkeypatch.setattr(results, "_state", counted)
    assert result.point_margin_summary(bootstrap_draws=2)["games"].tolist() == [200]
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change", [{"rng_protocol": None}, {"rng_protocol": "Unknown"}, {"seed": 2**32}]
)
def test_unknown_random_coordinate_protocol_withholds_interval(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    root = _saved(tmp_path, [(0, 1, None), (0, 3, None)])
    path = root / "run_details.json"
    content = json.loads(path.read_text())
    for entry in content["passes"].values():
        entry["details"].update(change)
    path.write_text(json.dumps(content))
    columns = load_results(root).point_margin_summary(bootstrap_draws=20)
    assert np.isnan(columns["ci_low"][0]) and columns["independent_units"][0] is None
