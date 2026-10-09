"""Check that the committed hyperparameter search study 1 prepares as declared.

prepare_declared_study binds scripts/paper_experiments/hyperparameter_search/
shapes_and_ppo.yaml to the local opponent bundle. Contracts checked here: the
prepared study previews as 54 runs, 18 cases of three search seeds that
include 12259871 (the seed of the pooled 250M model), with unique
method-prefixed case IDs and opponent shares that sum to one; QMIX and
PQN-VDN use the team reward and PPO methods the per-agent reward; every run
shares validation maps 42, 43, 44 and 46; the validation panel binds the 250M
model's checked folder first, then Alpha, Beta and Gamma; the grouping file
passes the analysis owner with its ten recipe picks; and a changed expected
identity or a reordered validation panel is refused before protocol/ is
written. The bundle lives in ignored artifacts/, so these checks skip where
it is absent. No game, learning update or GPU work runs here.
"""

from pathlib import Path

import pytest
import yaml

from marl_battlegrounds import training
from marl_battlegrounds.experiments.prepare import prepare_declared_study
from marl_battlegrounds.training.analysis import (
    _training_grouping,  # pyright: ignore[reportPrivateUsage]
)

ROOT = Path(__file__).resolve().parents[2]
SEARCH = ROOT / "scripts/paper_experiments/hyperparameter_search"
BUNDLE = ROOT / "artifacts/paper-experiments/hyperparameter-search/opponents"
POOLED = "recurrent-mappo-combat-support-pooled-step-250675200"

pytestmark = pytest.mark.skipif(
    not (BUNDLE / "pool.json").is_file(), reason="local opponent bundle is absent"
)


def _prepare(tmp_path: Path, **changes: Path) -> Path:
    return prepare_declared_study(
        changes.get("declaration", SEARCH / "shapes_and_ppo.yaml"),
        BUNDLE,
        tmp_path / "out",
        expected_opponents=changes.get("expected", SEARCH / "opponents.yaml"),
        recipe_selection=SEARCH / "shapes_and_ppo_picks.yaml",
    )


def test_study_one_prepares_as_declared(tmp_path: Path) -> None:
    study_file = _prepare(tmp_path)
    protocol = study_file.parent
    study = yaml.safe_load(study_file.read_text())
    preview = training.preview_study(study, output_dir=tmp_path / "out/study")
    runs = preview["cases"]
    assert len(runs) == 54
    assert len({run["id"] for run in runs}) == 54
    assert {run["config"]["seed"] for run in runs} == {12259871, 20261001, 20261002}
    for run in runs:
        config = run["config"]
        assert run["id"].startswith(config["method"] + "-")
        assert sum(config["opponents"].values()) == pytest.approx(1.0)
        assert config["validation_maps"] == [42, 43, 44, 46]
        assert config["retain_all_actors"] is True
        team = config["method"] in ("qmix", "pqn_vdn")
        assert config["reward"].endswith(
            "combat_support_team_reward" if team else "combat_support_reward"
        )
        panel = run["runtime"]["validation_opponents"]
        assert Path(panel[0]).name == POOLED
        assert panel[1:] == ["tdm-alpha", "tdm-beta", "tdm-gamma"]
    grouping = _training_grouping(protocol / "grouping.yaml", None)
    assert len(grouping["runs"]) == 54
    assert len(grouping["recipe_selection"]) == 10


def test_changed_identity_or_panel_order_is_refused(tmp_path: Path) -> None:
    expected = yaml.safe_load((SEARCH / "opponents.yaml").read_text())
    expected["members"][0]["inference_digest"] = "0" * 64
    changed = tmp_path / "opponents.yaml"
    changed.write_text(yaml.safe_dump(expected))
    with pytest.raises(ValueError):
        _prepare(tmp_path, expected=changed)
    declaration = (SEARCH / "shapes_and_ppo.yaml").read_text()
    moved = f"        - {POOLED}\n"
    assert declaration.count(moved) == 1
    reordered = declaration.replace(moved, "").replace(
        "        - tdm-gamma\n", "        - tdm-gamma\n" + moved
    )
    other = tmp_path / "reordered" / "shapes_and_ppo.yaml"
    other.parent.mkdir()
    other.write_text(reordered)
    with pytest.raises(ValueError, match="in order"):
        _prepare(tmp_path, declaration=other, expected=SEARCH / "opponents.yaml")
    assert not (tmp_path / "out/protocol").exists()
