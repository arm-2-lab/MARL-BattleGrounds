"""Check that Core config checks are repeated only when the exact inputs change.

run_config_check remembers a passed check by its check function and the
config's exact tree, leaf types, dtypes, shapes, weak types and bytes. The
same config is checked once; a changed byte or another check function runs the
real check again. A failure is never remembered and re-raises every time.
Traced configs and non-config values always reach the check, and the real
Core validator still rejects a changed invalid config.
"""

# pyright: reportPrivateUsage=false
import jax
import jax.numpy as jnp
import pytest

from marl_battlegrounds._config_checks import run_config_check
from marl_battlegrounds.core.config import validate_env_config
from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.tasks import make_canonical_team_deathmatch_evaluation_config


def _config() -> EnvConfig:
    return make_canonical_team_deathmatch_evaluation_config(map_id=47)


def _copy_arrays(value: object) -> object:
    return jnp.array(value) if isinstance(value, jax.Array) else value


def test_identical_config_is_checked_once_and_changes_run_again() -> None:
    calls: list[EnvConfig] = []

    def check(config: EnvConfig) -> None:
        calls.append(config)

    config = _config()
    run_config_check(check, config)
    run_config_check(check, config)
    run_config_check(check, jax.tree.map(_copy_arrays, config))
    assert len(calls) == 1
    changed = config._replace(max_steps=config.max_steps + 1)
    run_config_check(check, changed)
    assert len(calls) == 2
    weak = config._replace(max_steps=jnp.asarray(299))
    strong = config._replace(max_steps=jnp.asarray(299, jnp.int32))
    run_config_check(check, weak)
    run_config_check(check, strong)
    assert len(calls) == 4

    def other(config: EnvConfig) -> None:
        calls.append(config)

    run_config_check(other, config)
    assert len(calls) == 5


def test_failures_are_never_remembered() -> None:
    attempts: list[int] = []

    def reject(config: EnvConfig) -> None:
        attempts.append(1)
        raise ValueError("rejected by the check")

    config = _config()
    for _ in range(2):
        with pytest.raises(ValueError, match="rejected by the check"):
            run_config_check(reject, config)
    assert len(attempts) == 2


def test_traced_and_non_config_values_always_reach_the_check() -> None:
    calls: list[object] = []

    def check(config: object) -> None:
        calls.append(config)

    config = _config()

    for _ in range(2):
        # A fresh function each time, because make_jaxpr reuses earlier traces.
        def traced(steps: jax.Array) -> jax.Array:
            run_config_check(check, config._replace(max_steps=steps))
            return steps

        jax.make_jaxpr(traced)(jnp.int32(300))
    run_config_check(check, "not a config")
    run_config_check(check, "not a config")
    assert len(calls) == 4
    with pytest.raises(TypeError):
        run_config_check(validate_env_config, object())


def test_real_core_check_still_rejects_a_changed_invalid_config() -> None:
    config = _config()
    run_config_check(validate_env_config, config)
    invalid = config._replace(max_steps=0)
    for _ in range(2):
        with pytest.raises(ValueError):
            run_config_check(validate_env_config, invalid)
