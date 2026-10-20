"""Check the psro launcher's packaged settings and its call into training.

Contracts: the packaged psro settings compose to a recurrent IPPO oracle on
its 250,675,200-transition tournament recipe with the Season 0 validation
maps, the eight Season 0 training opponents as seeded members, the
four-member Season 0 validation panel, the released bundle (no folder given)
and train_psro's own defaults for every other population setting. run
resolves the oracle through the shared training parser, reads output_dir and
opponent_bundle from the launch folder, finds the seeded members and the
panel in the bundle by name, and passes every population setting to
train_psro unchanged, leaving its input settings as they were. A stand-in
replaces train_psro, so no game is played.
"""

import copy
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from marl_battlegrounds import training
from marl_battlegrounds.experiments import psro


def _composed(*overrides: str) -> dict[str, Any]:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        config = compose(config_name="psro", overrides=list(overrides))
    return OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportReturnType]


def test_settings_compose() -> None:
    config = _composed("output_dir=o")
    assert (config["method"], config["total_env_steps"]) == ("ippo", 250675200)
    assert config["validation_maps"] == [42, 43, 44, 46]
    settings = config["psro"]
    assert len(settings["seed_members"]) == 8
    assert settings["seed_members"][-2:] == ["tdm-alpha", "tdm-beta"]
    assert settings["panel"] == [
        "tdm-alpha",
        "tdm-beta",
        "tdm-gamma",
        "recurrent-mappo-combat-support-pooled-step-250675200",
    ]
    assert settings.pop("opponent_bundle") is None
    del settings["seed_members"], settings["panel"]
    defaults = inspect.signature(training.train_psro).parameters
    assert settings == {name: defaults[name].default for name in settings} | {
        "payoff_maps": list(range(42))
    }


def test_run_resolves_and_calls_train_psro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "models" / "fixed").mkdir(parents=True)
    members = [
        {"name": "fixed", "kind": "actor", "path": "models/fixed"},
        {"name": "tdm-alpha", "kind": "scripted"},
        {"name": "tdm-beta", "kind": "scripted"},
    ]
    (bundle / "pool.json").write_text(
        json.dumps({"schema_version": 1, "members": members}), encoding="utf-8"
    )
    settings = _composed(
        "recipe=software_check",
        "output_dir=run",
        "psro.opponent_bundle=bundle",
        "psro.seed_members=[fixed,tdm-alpha]",
        "psro.panel=[tdm-beta]",
        "psro.payoff_games=6",
    )
    before = copy.deepcopy(settings)
    calls: list[tuple[training.TrainConfig, dict[str, object]]] = []

    def stand_in(config: training.TrainConfig, **kwargs: object) -> dict[str, Any]:
        calls.append((config, kwargs))
        return {"entrant": None}

    monkeypatch.setattr(training, "train_psro", stand_in)
    assert psro.run(settings, base_dir=tmp_path) == {"entrant": None}
    assert settings == before
    [(config, kwargs)] = calls
    assert (config.method, config.seed, config.total_env_steps) == ("ippo", 710, 20480)
    assert kwargs.pop("output_dir") == tmp_path / "run"
    assert kwargs.pop("seed_members") == {
        "fixed": str((bundle / "models" / "fixed").resolve()),
        "tdm-alpha": "tdm-alpha",
    }
    assert kwargs.pop("panel") == {"tdm-beta": "tdm-beta"}
    assert kwargs == {
        key: value
        for key, value in before["psro"].items()
        if key not in {"seed_members", "panel", "opponent_bundle"}
    }
    assert kwargs["payoff_games"] == 6
