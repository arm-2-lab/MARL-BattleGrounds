"""Remember Core config checks that passed, keyed by the exact config bytes.

Core's validate_env_config and validate_product_env_config are pure host checks
of one EnvConfig: they return None or raise. Training setup, checkpoint restore
and evaluation run them many times on byte-identical configs (52 maps, their
exchanged spawn banks and 8 scenarios, then all again at restore). Each check
walks every spawn pad and obstacle with small eager device calls, so the
repeats cost tens of seconds per training start, and far more on a busy host.

run_config_check runs the real Core check the first time a config is seen and
remembers only a success. The key is the check function itself plus the
config's tree layout and, for every leaf, its Python type, dtype, shape, weak
type and bytes. A changed byte, or a different check function (including a
test's replacement), runs the real check again. A failure is never
remembered, so it raises Core's own error every time. Traced configs are
passed straight to Core. Core itself is unchanged.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from typing import Any, cast

import jax
import numpy as np
from jax.core import Tracer

from marl_battlegrounds.core.types import EnvConfig

_PASSED: set[tuple[Callable[[EnvConfig], None], str]] = set()
_LOCK = threading.Lock()
_LIMIT = 8192


def _config_key(config: EnvConfig) -> str | None:
    """Describe one concrete config exactly, or None when it holds a tracer.

    Parameters
    ----------
    config : EnvConfig
        A scalar or batched config whose leaves are JAX arrays, NumPy arrays
        or Python scalars, on any device.

    Returns
    -------
    str or None
        A SHA-256 hex digest of the tree layout and, for each leaf, its Python
        type name, dtype, shape, weak type and raw bytes. None when any leaf is
        a tracer, because a traced value has no bytes to compare.

    Notes
    -----
    Host-only. Python values and CPU arrays are read in place; if any leaf
    lives on an accelerator, all leaves come back in one device_get call. The
    device a leaf lives on is not part of the key: Core's checks read values,
    and the same values pass or fail the same way.
    """
    leaves, tree = cast(tuple[list[Any], Any], jax.tree.flatten(config))
    if any(isinstance(leaf, Tracer) for leaf in leaves):
        return None
    on_device = any(
        isinstance(leaf, jax.Array)
        and any(device.platform != "cpu" for device in leaf.devices())
        for leaf in leaves
    )
    # Host and CPU values are read in place; device values come back in one copy.
    hosts = cast(list[Any], jax.device_get(leaves)) if on_device else leaves
    digest = hashlib.sha256(str(tree).encode())
    for leaf, host in zip(leaves, hosts, strict=True):
        value = np.ascontiguousarray(np.asarray(host))
        weak = bool(getattr(leaf, "weak_type", False))
        digest.update(
            repr((type(leaf).__name__, value.dtype.str, value.shape, weak)).encode()
        )
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_config_check(check: Callable[[EnvConfig], None], config: Any) -> None:  # noqa: ANN401
    """Run a Core config check once per exact config and remember a success.

    Parameters
    ----------
    check : callable
        validate_env_config, validate_product_env_config, or a caller's
        replacement with the same contract: return None or raise.
    config : Any
        The value to check. Anything other than a concrete EnvConfig goes
        straight to check, so Core raises its own TypeError.

    Returns
    -------
    None
        The config passed, now or on an earlier call with identical bytes.

    Raises
    ------
    TypeError, ValueError
        Whatever check raises. Failures are never remembered.

    Notes
    -----
    Host-only and thread-safe. The remembered set holds at most 8,192 keys
    and is cleared when full, which only costs repeated checks. Keys are
    digests; no config arrays are kept alive.
    """
    key = _config_key(config) if isinstance(config, EnvConfig) else None
    if key is None:
        check(config)
        return
    with _LOCK:
        if (check, key) in _PASSED:
            return
    check(config)
    with _LOCK:
        if len(_PASSED) >= _LIMIT:
            _PASSED.clear()
        _PASSED.add((check, key))
