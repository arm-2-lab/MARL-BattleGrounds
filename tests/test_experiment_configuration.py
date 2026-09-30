"""Check packaged settings and previews against the existing public config API.

Cover all original fields, six method groups, complete pool replacement, ordinary
Hydra overrides and rejection paths. Subprocess probes forbid numerical imports,
factory calls and workers. These are software checks, not learning experiments.
"""

import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from marl_battlegrounds.baselines.config import PPOConfig, PQNConfig, QMIXConfig
from marl_battlegrounds.baselines.methods import TRAINING_METHODS, method_settings_field
from marl_battlegrounds.experiments.hydra import resolve_training_config, run_training
from marl_battlegrounds.training.config import TrainConfig, config_to_dict

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    (record.__name__, item.name)
    for record in (TrainConfig, PPOConfig, QMIXConfig, PQNConfig)
    for item in fields(record)
]


def composed(*overrides: str) -> DictConfig:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        return compose(config_name="train", overrides=list(overrides))


@pytest.mark.parametrize("method", TRAINING_METHODS)
def test_default_recipe_matches_original_api_defaults(
    method: str, tmp_path: Path
) -> None:
    actual = resolve_training_config(composed(f"method={method}"), base_dir=tmp_path)
    expected = json.loads(
        (ROOT / "tests/fixtures/training_config_defaults.json").read_text()
    )
    assert config_to_dict(actual) == {**expected[method], "retain_all_actors": False}


@pytest.mark.parametrize(("record", "field"), FIELDS)
def test_every_api_field_is_available(record: str, field: str, tmp_path: Path) -> None:
    blocks = {
        "PPOConfig": ("mappo", "ppo"),
        "QMIXConfig": ("qmix", "qmix"),
        "PQNConfig": ("pqn_vdn", "pqn"),
    }
    method, block = blocks.get(record, ("mappo", None))
    cfg = composed(f"method={method}")
    effective = config_to_dict(resolve_training_config(cfg, base_dir=tmp_path))
    if block:
        assert field in cfg[block]
        assert cfg[block][field] == effective[block][field]
    elif field in ("qmix", "pqn"):
        selected = composed(f"method={'qmix' if field == 'qmix' else 'pqn_vdn'}")
        assert field in selected
    else:
        assert field in cfg
        assert field in effective


def test_original_census_and_generated_presets_do_not_drift() -> None:
    assert len(FIELDS) == 89  # Original 88 fields plus opt-in actor retention.
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/dev/generate_experiment_configs.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("before", TRAINING_METHODS)
@pytest.mark.parametrize("after", TRAINING_METHODS)
def test_method_switch_keeps_only_selected_settings(
    before: str, after: str, tmp_path: Path
) -> None:
    cfg = composed(f"method={before}", f"method={after}", "recipe=software_check")
    result = resolve_training_config(cfg, base_dir=tmp_path)
    assert result.method == after
    assert set(cfg) & {"ppo", "qmix", "pqn"} == {method_settings_field(after)}
    assert result.total_env_steps == 20480
    assert result.num_envs == 32
    assert result.retain_all_actors is True


@pytest.mark.parametrize("method", TRAINING_METHODS)
def test_explicit_wrong_method_block_is_rejected(method: str, tmp_path: Path) -> None:
    wrong = "qmix" if method_settings_field(method) == "ppo" else "ppo"
    with pytest.raises(ValueError, match=f"takes no {wrong}"):
        resolve_training_config(
            composed(f"method={method}", f"+{wrong}={{}}"), base_dir=tmp_path
        )


def test_pool_and_curriculum_groups_are_complete_declarations(tmp_path: Path) -> None:
    cfg = composed("pool=scripted", "curriculum_preset=staged", "resources=cpu")
    result = resolve_training_config(cfg, base_dir=tmp_path)
    assert result.opponents == {
        "self": 0.6,
        "past": 0.2,
        "tdm-alpha": 1 / 15,
        "tdm-beta": 1 / 15,
        "tdm-gamma": 1 / 15,
    }
    assert result.curriculum is True
    reset = composed(
        "pool=scripted",
        "pool=none",
        "curriculum_preset=staged",
        "curriculum_preset=none",
        "resources=cpu",
        "resources=inherit",
    )
    clean = resolve_training_config(reset, base_dir=tmp_path)
    assert clean.opponents is None
    assert clean.curriculum is False
    assert reset.resources is None
    with pytest.raises(ValueError, match="either a named pool or inline"):
        resolve_training_config(
            composed("pool=scripted", "opponents={self:1}"), base_dir=tmp_path
        )


def test_nested_settings_interpolation_lists_and_quoted_paths(tmp_path: Path) -> None:
    cfg = composed(
        'validation_panel="inputs/frozen panel.json"',
        "seed=7",
        "ppo.actor_lr=0.0003",
        "ppo.critic_lr=${ppo.actor_lr}",
        "opponents=[self,past,tdm-alpha]",
        "curriculum=[{share:1.0,maps:[0,1],team_size:5}]",
    )
    result = resolve_training_config(cfg, base_dir=tmp_path)
    assert result.validation_panel == str(tmp_path / "inputs/frozen panel.json")
    assert result.ppo.actor_lr == result.ppo.critic_lr == 0.0003
    assert result.opponents == ("self", "past", "tdm-alpha")
    assert result.curriculum is not False
    null = resolve_training_config(composed("validation_panel=null"), base_dir=tmp_path)
    assert null.validation_panel is None


@pytest.mark.parametrize(
    "setting",
    [
        "num_envs=true",
        "+typo=1",
        "ppo.actor_lr=.nan",
        "ppo.gamma=.inf",
        "seed=???",
        "ppo.actor_lr=${missing}",
    ],
)
def test_invalid_settings_fail_before_any_output(setting: str, tmp_path: Path) -> None:
    with pytest.raises((ValueError, TypeError, KeyError, OmegaConfBaseException)):
        run_training(
            composed("preview=true", setting),
            base_dir=tmp_path,
            launch_dir=tmp_path / "launch",
        )
    assert not list(tmp_path.iterdir())


def test_non_string_keys_and_object_values_are_not_coerced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="string mapping keys"):
        resolve_training_config(cast(Any, {1: "bad"}), base_dir=tmp_path)
    with pytest.raises((TypeError, ValueError)):
        resolve_training_config({"seed": object()}, base_dir=tmp_path)


def test_preview_has_effective_settings_and_exact_paths(tmp_path: Path) -> None:
    cfg = composed(
        "preview=true",
        "method=qmix",
        "recipe=software_check",
        'output_dir="my run"',
        "resources=cpu",
    )
    report = run_training(cfg, base_dir=tmp_path, launch_dir=tmp_path / "launch")
    assert isinstance(report, dict)
    assert report["case_count"] == 1
    assert report["config"]["qmix"]["rollout_length"] == 4
    assert report["total_env_steps"] == 20480
    assert report["output_dir"] == str(tmp_path / "my run")
    assert report["launch_dir"] == str(tmp_path / "launch")
    assert report["resources"] == {"devices": [{"device": "cpu", "max_workers": 1}]}
    assert report["pending_checks"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["--cfg", "job", "--resolve", "method=qmix"],
        ["preview=true", "method=pqn_vdn", "recipe=software_check"],
    ],
)
def test_public_entry_is_backend_free_and_starts_no_workers(
    tmp_path: Path, arguments: list[str]
) -> None:
    program = r"""import sys, runpy, subprocess
from importlib.abc import MetaPathFinder
class BlockNumerical(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'jax', 'jaxlib', 'numpy', 'flax', 'optax', 'flashbax'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Numerical import: ' + fullname)
sys.meta_path.insert(0, BlockNumerical())
popen = subprocess.Popen
def forbidden(*args, **kwargs):
    command = args[0] if args else kwargs.get('args')
    if (isinstance(command, (list, tuple)) and len(command) >= 4
            and command[0] == 'git' and command[1] == '-C'
            and command[3] in {'check-ignore', 'ls-files'}):
        return popen(*args, **kwargs)
    raise AssertionError(f'A worker or factory started: {command!r}')
subprocess.Popen = forbidden
sys.argv = ['train', *sys.argv[1:]]
runpy.run_module('marl_battlegrounds.experiments.train', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, *arguments],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if arguments[0] == "preview=true":
        report = json.loads(result.stdout)
        assert not Path(report["output_dir"]).exists()
        assert Path(report["launch_dir"], ".hydra/config.yaml").is_file()
    else:
        assert not list(tmp_path.iterdir())


def test_method_group_disagreement_is_rejected_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra.core.hydra_config import HydraConfig

    monkeypatch.setattr(HydraConfig, "initialized", staticmethod(lambda: True))
    selected = OmegaConf.create({"runtime": {"choices": {"method": "qmix"}}})
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: selected))
    with pytest.raises(ValueError, match="selected method group"):
        run_training(
            composed("preview=true"), base_dir=tmp_path, launch_dir=tmp_path / "launch"
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("invalid", [{"random": 99}, "random", True, 1])
def test_validation_opponent_container_rejection_matches_api(
    invalid: object, tmp_path: Path
) -> None:
    from marl_battlegrounds.training.config import config_from_dict

    with pytest.raises(TypeError, match="JSON array"):
        config_from_dict({"validation_opponents": invalid})
    with pytest.raises(TypeError, match="JSON array"):
        resolve_training_config({"validation_opponents": invalid}, base_dir=tmp_path)


def test_factory_reference_stays_a_reference_despite_same_named_folder(
    tmp_path: Path,
) -> None:
    (tmp_path / "custom:factory").mkdir()
    config = resolve_training_config(
        {"validation_opponents": ["custom:factory"]}, base_dir=tmp_path
    )
    assert config.validation_opponents == ("custom:factory",)


@pytest.mark.parametrize("preview", ["true", "false"])
def test_multirun_refuses_the_basic_launcher(tmp_path: Path, preview: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "marl_battlegrounds.experiments.train",
            "--multirun",
            "hydra/launcher=basic",
            f"preview={preview}",
            "seed=1,2",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "require the study launcher" in result.stdout + result.stderr
    assert "Launching 2 jobs" not in result.stdout + result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", TRAINING_METHODS)
def test_eleven_member_pool_and_exact_validator_keep_ordinary_bindings(
    method: str,
    tmp_path: Path,
) -> None:
    names = (
        *(f"saved-actor-{index}" for index in range(8)),
        "tdm-alpha",
        "tdm-beta",
        "tdm-gamma",
    )
    shares = {"self": 0.6, "past": 0.2, **dict.fromkeys(names, 0.2 / 11)}
    bindings = {name: f"models/{name}" for name in names[:8]}
    value = composed(
        f"method={method}",
        "opponents={"
        + ",".join(f"{key}:{value}" for key, value in shares.items())
        + "}",
        "+runtime.opponents={"
        + ",".join(f"{key}:{value}" for key, value in bindings.items())
        + "}",
        f"+runtime.validation_opponents=[{bindings[names[6]]}]",
        "preview=true",
    )
    result = run_training(
        value,
        base_dir=tmp_path,
        launch_dir=tmp_path / "launch",
    )
    assert isinstance(result, dict)
    assert result["config"]["opponents"] == shares
    assert list(result["config"]["opponents"]) == list(shares)
    assert result["runtime"]["validation_opponents"] == [
        str(tmp_path / bindings[names[6]])
    ]
    assert (
        result["runtime"]["opponents"][names[6]]
        == result["runtime"]["validation_opponents"][0]
    )
    assert not (tmp_path / "launch").exists()
