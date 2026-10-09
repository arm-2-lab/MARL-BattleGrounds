"""Check shared scientific conditions, live confirmation needs and lazy helpers.

Host-only fixtures exercise comparisons without providers, model restores or
new games. The separate System-validation test executes a real partner-trained
request through the public validator and a linked saved selection decision.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.training.test_training_selection import _evidence, _reader

from marl_battlegrounds.training import analysis, selection, validation

# pyright: reportPrivateUsage=false

CHANGES = [
    {"panel_digest": "other"},
    {"maps": [43]},
    {"system_roster": ["mage", "priest"]},
    {"opponent_roster": ["warrior"]},
    {"learner_slots": [0]},
    {"red_zone_depth": 0.0},
    {"selection_schema_version": 1},
    {"spawn_mode": "fixed"},
    {"selection_rule": "other"},
    {"cell_weights": [0.2, 0.8]},
    {"partners": [{"name": "Ally", "label": "held_out", "registration_id": "one"}]},
]


@pytest.mark.parametrize("changed", CHANGES)
def test_conditions_refuse_changes_before_confirmation_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: dict[str, Any]
) -> None:
    roots = [tmp_path / "a", tmp_path / "b"]
    items = {root: _evidence(root, index) for index, root in enumerate(roots)}
    for row in items[roots[1]]["records"]:
        row.update(changed)
    _reader(monkeypatch, items)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Incompatible conditions reached a confirmation request")

    monkeypatch.setattr(selection, "_request", forbidden)
    with pytest.raises(ValueError, match="Selection cannot mix"):
        selection.reselect_checkpoint(
            roots,
            declaration={"name": "Compare", "across_runs": True, "seed_order": [0, 1]},
            output_dir=tmp_path / "choice",
        )
    assert not (tmp_path / "choice").exists()
    first, second = items[roots[0]]["records"][1], items[roots[1]]["records"][1]
    with pytest.raises(ValueError, match="Selection cannot mix"):
        analysis.select_checkpoint([first, dict(second, checkpoint_id="another")])


def test_partner_identity_labels_and_order_are_conditions() -> None:
    original = {
        "partners": [
            {"name": "One", "label": "familiar", "registration_id": "a"},
            {"name": "Two", "label": "held_out", "registration_id": "b"},
        ]
    }
    for key in ("name", "label", "registration_id"):
        changed = copy.deepcopy(original)
        changed["partners"][0][key] = "changed"
        assert validation.validation_conditions(
            original
        ) != validation.validation_conditions(changed)
    assert validation.validation_conditions(
        original
    ) != validation.validation_conditions(
        {"partners": list(reversed(original["partners"]))}
    )
    assert validation.validation_conditions(
        {"seed": 1, "root": 10}
    ) == validation.validation_conditions({"seed": 2, "root": 20})


def test_live_panel_request_is_explicit_json_without_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _evidence(tmp_path / "run")
    item["records"] = [row for row in item["records"] if row["purpose"] == "routine"]
    _reader(monkeypatch, {tmp_path / "run": item})

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Selection loaded an executable method")

    monkeypatch.setattr(validation, "load_panel", forbidden)
    result = selection.reselect_checkpoint(
        [tmp_path / "run"], declaration={"name": "Live"}, output_dir=tmp_path / "choice"
    )
    for request in result["runs"][0]["needs_confirmation"]:
        assert not request["ready_to_run"]
        assert request["required_bindings"]["panel"] == [
            {"name": "Opponent", "registration_id": "opponent"}
        ]
        assert request["arguments"]["expected_task"] == request["expected_task"]
        json.dumps(request, allow_nan=False)
        compile(request["python"], "confirmation", "exec")


def test_validation_facade_is_lazy_and_uses_existing_owners() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; from marl_battlegrounds import training; "
            "assert 'jax' not in sys.modules; "
            "assert all(name in training.__all__ for name in "
            "('create_panel','load_panel','validate_checkpoint'))",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    from marl_battlegrounds import training

    for name in ("create_panel", "load_panel", "validate_checkpoint"):
        assert getattr(training, name) is getattr(validation, name)


@pytest.mark.parametrize("changed", CHANGES)
def test_grouped_reports_keep_incompatible_rows_without_pooled_estimates(
    tmp_path: Path, changed: dict[str, Any]
) -> None:
    from tests.training.test_training_uncertainty import _grouping, _reduce, _run

    first = _run(tmp_path / "first", (0.0, 1.0, 0.0, 1.0))
    second = _run(tmp_path / "second", (1.0, 0.0, 1.0, 0.0))
    original = json.loads((second / "selection.json").read_text())
    # The fixture's old selection schema defaults to one.
    if changed == {"selection_schema_version": 1}:
        changed = {"selection_schema_version": 2}
    if "selection_rule" in changed:
        path = second / "run_details.json"
        details = json.loads(path.read_text())
        details.update(changed)
        path.write_text(json.dumps(details))
    else:
        original.update(changed)
        (second / "selection.json").write_text(json.dumps(original))
    report = _reduce(_grouping([(first, "Method", "one"), (second, "Method", "two")]))
    assert len(report["runs"]) == 2
    assert all(len(row["cells"]) == 4 for row in report["runs"])
    group = report["groups"][0]
    assert group["score"] is group["ci_low"] is group["ci_high"] is None
    assert group["unavailable_reasons"]


def test_plain_request_keeps_the_original_noncustom_task(tmp_path: Path) -> None:
    item = _evidence(tmp_path)
    request = selection._request(
        item,
        item["records"][0],
        selection._declaration({"name": "Same"}),
        tmp_path / "choice",
    )
    assert "maps" not in request["arguments"]
    assert "system_roster" not in request["arguments"]
    assert (
        request["expected_task"]["schema_version"]
        == item["records"][0]["schema_version"]
    )


def test_validation_maps_default_checks_and_old_saved_configs() -> None:
    from marl_battlegrounds.training.config import (
        TrainConfig,
        config_from_dict,
        config_to_dict,
    )
    from marl_battlegrounds.training.runner import _saved_config

    default = TrainConfig()
    assert default.validation_maps is None
    assert validation.VALIDATION_MAPS == (42, 43, 44, 45, 46)
    assert "validation_maps" not in config_to_dict(default)
    # A list, as YAML gives it; the field's documented contract accepts lists.
    chosen = TrainConfig(validation_maps=[46, 42, 44, 43])  # pyright: ignore[reportArgumentType]
    assert chosen.validation_maps == (42, 43, 44, 46)
    assert config_to_dict(chosen)["validation_maps"] == [42, 43, 44, 46]
    assert config_from_dict(config_to_dict(chosen)) == chosen
    for bad in ([], [42, 42], [41], [47], [True], [42.0], "42", {42}, (45, 45)):
        with pytest.raises(ValueError, match="validation_maps"):
            TrainConfig(validation_maps=bad)  # pyright: ignore[reportArgumentType]
    # Saved configs from before the setting have no key and read as the default.
    fixture = Path(__file__).parents[1] / "fixtures/training_config_defaults.json"
    for saved in json.loads(fixture.read_text()).values():
        assert "validation_maps" not in saved
        restored = config_from_dict(saved)
        assert restored.validation_maps is None
        assert config_to_dict(restored) == {**saved, "retain_all_actors": False}
        old_run = _saved_config({"metadata": {"config": dict(saved)}})
        assert config_to_dict(old_run) == config_to_dict(restored)


def test_four_validation_maps_are_saved_and_used_by_every_validation_game(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import runner

    maps = [42, 43, 44, 46]
    config = runner.TrainConfig(
        method="ff_ippo",
        # Eight lanes play each validation pass's eight games in one batch.
        num_envs=8,
        total_env_steps=16,
        keep_past=0,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        curriculum=[{"share": 1.0, "maps": [0], "team_size": 1}],
        validation_opponents=("random",),
        validation_maps=(46, 42, 43, 44),
        validation_fractions=(1.0,),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
        metrics="none",
        verbose=False,
    )
    result = runner.train(config, output_dir=tmp_path / "run")
    details = json.loads((result.run_dir / "run_details.json").read_text())
    assert details["config"]["validation_maps"] == maps
    assert details["validation_deployment"]["maps"] == maps
    assert details["validation_declaration"]["deployment"]["maps"] == maps
    records = json.loads((result.run_dir / "validation_results.json").read_text())
    assert [row["purpose"] for row in records] == [
        "routine",
        "routine",
        "confirmation",
    ]
    for row in records:
        assert row["maps"] == maps
        assert sorted(cell["map_id"] for cell in row["cells"]) == maps
        assert row["games"] == 8
        assert validation.validation_conditions(row)["maps"] == maps
    selected = json.loads((result.run_dir / "selection.json").read_text())
    assert selected["purpose"] == "confirmation" and selected["maps"] == maps
    # A resume keeps the saved maps; a resume config naming other maps fails.
    resumed = runner.train(resume_from=result.final_checkpoint)
    assert resumed.selected_actor == result.selected_actor
    with pytest.raises(ValueError, match="Resume config differs"):
        runner.train(
            replace(config, validation_maps=None), resume_from=result.final_checkpoint
        )
