"""Check backend-free configuration, old aliases and unchanged saved defaults.

Fresh processes keep these checks independent of pytest's numerical fixtures.
The public export list must include every registered lazy helper without
importing those helpers or their optional dependencies.
The JSON fixture was captured before the configuration extraction, for the
six methods that existed then; HAPPO's defaults are recurrent MAPPO's with
its own method name and one actor per class, and a custom HAPPO ppo that
shares one actor is refused. Numerical module aliases are checked separately,
since those modules require training.
"""

import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from marl_battlegrounds.baselines.config import PPOConfig, PQNConfig, QMIXConfig
from marl_battlegrounds.baselines.methods import TRAINING_METHODS
from marl_battlegrounds.training.config import (
    TrainConfig,
    config_from_dict,
    config_to_dict,
)


def test_every_method_and_full_host_validation_without_numerical_imports() -> None:
    script = """
import sys
from importlib.abc import MetaPathFinder
class NoNumericalImports(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'jax', 'jaxlib', 'flax', 'optax', 'flashbax', 'numpy'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoNumericalImports())
from marl_battlegrounds import training
assert len(training.__all__) == len(set(training.__all__))
assert set(training.__all__) == set(training._OWNERS)
from marl_battlegrounds.training import TrainConfig
from marl_battlegrounds.training.config import config_from_dict, config_to_dict
from marl_battlegrounds.baselines.methods import TRAINING_METHODS
from marl_battlegrounds.baselines import config
from marl_battlegrounds.training.study import _config
from marl_battlegrounds.training.search import declaration
from marl_battlegrounds._task_settings import canonical_tournament_rosters
roster = ("mage", "warrior", "hunter", "rogue", "priest")
assert canonical_tournament_rosters() == (roster, roster)
from pathlib import Path
for method in TRAINING_METHODS:
    c = TrainConfig(method=method, curriculum=[{'share': 1, 'maps': [0, 1],
        'rosters': {'system': ['priest', 'priest'], 'opponent': ['hunter']}}],
        opponents={'self': .6, 'past': .2, 'tdm-alpha': .2},
        partners=['tdm-beta'],
        validation_opponents=('tdm-gamma', '/tmp/actor', 'custom:factory'),
        pinned_opponent=None, history_capture_env_steps=(9600, 19200))
    assert config_from_dict(config_to_dict(c)) == c
    _config({'method': method, 'total_env_steps': 10000000,
             'validation_opponents': ['tdm-alpha', './actor']}, Path('/tmp'), 42)
assert declaration()
for change in ({'num_envs': 3}, {'method': 'unknown'},
               {'ppo': config.PPOConfig(gamma=True)},
               {'validation_opponents': ('relative-actor',)},
               {'opponents': {'self': -1}},
               {'curriculum': [{'share': 1, 'maps': [0],
                 'rosters': {'system': ['unknown'], 'opponent': ['mage']}}]},
               {'shaping_mode': 'unknown'}):
    try:
        TrainConfig(**change)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError(change)
assert not {'jax', 'flax', 'optax', 'flashbax', 'numpy'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        text=True,
        capture_output=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stderr


HISTORICAL_METHODS = ("mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn")


@pytest.mark.parametrize("method", HISTORICAL_METHODS)
def test_all_configuration_defaults_match_before_extraction(method: str) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "training_config_defaults.json"
    expected = json.loads(fixture.read_text())[method]
    actual = config_to_dict(config_from_dict({"method": method}))
    # Preserve the original fixture; this later opt-in leaves old fields intact.
    expected = {**expected, "retain_all_actors": False}
    assert actual == expected
    assert config_to_dict(config_from_dict(actual)) == expected


def test_happo_defaults_are_mappo_with_one_actor_per_class() -> None:
    assert set(TRAINING_METHODS) == {*HISTORICAL_METHODS, "happo"}
    mappo = config_to_dict(config_from_dict({"method": "mappo"}))
    happo = config_to_dict(config_from_dict({"method": "happo"}))
    assert happo == {
        **mappo,
        "method": "happo",
        "ppo": {**mappo["ppo"], "parameter_sharing": "class"},
    }
    assert config_from_dict(happo) == config_from_dict({"method": "happo"})
    custom = TrainConfig(method="happo", ppo=PPOConfig(parameter_sharing="none"))
    assert custom.ppo.parameter_sharing == "none"
    with pytest.raises(ValueError, match="one actor group at a time"):
        TrainConfig(method="happo", ppo=PPOConfig(actor_lr=0.0001))


def test_old_imports_are_the_same_objects() -> None:
    from marl_battlegrounds import _task_settings, tasks
    from marl_battlegrounds.baselines import config, ppo, pqn, qmix
    from marl_battlegrounds.training import TrainConfig as PublicConfig
    from marl_battlegrounds.training import runner

    assert (
        tasks.canonical_tournament_rosters
        is _task_settings.canonical_tournament_rosters
    )

    assert PublicConfig is runner.TrainConfig is TrainConfig
    assert runner.config_from_dict is config_from_dict
    assert runner.config_to_dict is config_to_dict
    for module, name in ((ppo, "PPO"), (qmix, "QMIX"), (pqn, "PQN")):
        assert getattr(module, name + "Config") is getattr(config, name + "Config")
        assert getattr(module, "DEFAULT_" + name + "_CONFIG") is getattr(
            config, "DEFAULT_" + name + "_CONFIG"
        )
        assert getattr(module, "validate_" + name.lower() + "_batch_size") is getattr(
            config, "validate_" + name.lower() + "_batch_size"
        )
    assert pqn.pqn_planned_learning_blocks is config.pqn_planned_learning_blocks


def test_learner_scalar_validation_keeps_its_existing_differences() -> None:
    assert PPOConfig(actor_lr=True).actor_lr is True
    for constructor in (QMIXConfig, PQNConfig):
        with pytest.raises(ValueError, match="q_lr"):
            constructor(q_lr=True)


@pytest.mark.parametrize("coefficient", [Fraction(1, 100), 2**64])
def test_config_preserves_legacy_scalar_rejection(coefficient: object) -> None:
    from typing import cast

    from marl_battlegrounds.training.shaping import validate_shaping

    with pytest.raises(TypeError, match="real scalar"):
        validate_shaping(discount=0.99, coefficient=cast(float, coefficient))
    with pytest.raises(TypeError, match="real scalar"):
        TrainConfig(shaping_coefficient=cast(float, coefficient))


@pytest.mark.parametrize("coefficient", [0, 1.0, 2**63 - 1, 2**63, 2**64 - 1])
def test_config_preserves_legacy_scalar_integer_boundary(coefficient: float) -> None:
    from marl_battlegrounds.training.shaping import validate_shaping

    validate_shaping(discount=0.99, coefficient=coefficient)
    config = TrainConfig(shaping_coefficient=coefficient)
    assert config_to_dict(config)["shaping_coefficient"] == coefficient
