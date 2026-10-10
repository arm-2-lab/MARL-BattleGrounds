"""Describe the actual software and device used by a recorded run.

Runtime capture is host-only and may initialize JAX's default backend. JAX is
imported inside the capture call so ordinary metadata readers do not initialize
numerical devices just by importing this module. Records contain versions and
device descriptions, not machine paths, measured speed or guessed driver data.
This module also owns the one reader of the declared XLA_FLAGS string and the
execution_settings record that says whether XLA's deterministic GPU mode was
declared for each attempt of a pass.
"""

from __future__ import annotations

import os
import platform as host_platform
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Protocol, cast

from marl_battlegrounds.evaluation._replays.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.models import CodeRevisionV1


class _RuntimeClient(Protocol):
    """Minimum runtime-client fields read when describing an active JAX backend."""

    platform_version: str


class _RuntimeDevice(Protocol):
    """Minimum device fields needed to record backend hardware and runtime version."""

    device_kind: str
    client: _RuntimeClient


def capture_debugger_runtime_provenance_v1(
    code_revision: CodeRevisionV1,
    *,
    policy_execution_included: bool = False,
) -> RuntimeProvenanceV1:
    """Capture one path-free host/runtime record for a recording launch.

    Imports of numerical runtimes stay inside this recording-only call so an
    ordinary CLI parse or a read-only replay launch does not acquire a new
    simulator/runtime dependency.

    Parameters
    ----------
    code_revision : CodeRevisionV1
        Exact validated CodeRevisionV1 for this recording launch.
    policy_execution_included : bool
        Whether policy work belongs to the record,
        default False for manual debugger stepping.

    Returns
    -------
    RuntimeProvenanceV1
        RuntimeProvenanceV1 for one environment on the actual default JAX backend.

    Raises
    ------
    TypeError
        code_revision has another type or the inclusion flag is not bool.
    RuntimeError
        JAX exposes no selected device or invalid runtime metadata.

    This host-only call may initialize JAX devices. It reads installed package
    versions and returns metadata; it writes no files or local paths.
    """
    if type(code_revision) is not CodeRevisionV1:
        raise TypeError("code_revision must be exact CodeRevisionV1")
    if type(policy_execution_included) is not bool:
        raise TypeError("policy_execution_included must be an exact bool")
    return capture_runtime_provenance(
        code_revision.package_version,
        policy_execution_included=policy_execution_included,
    )


def capture_runtime_provenance(
    package_version: str,
    *,
    policy_execution_included: bool = True,
    num_envs: int = 1,
) -> RuntimeProvenanceV1:
    """Describe the active numerical runtime for one evaluation pass.

    Parameters
    ----------
    package_version : str
        Version of the MARL-BGs source/package being recorded.
    policy_execution_included : bool
        Whether policy execution is included in this
        run's declared work; default True.
    num_envs : int
        Environment count recorded in batch_shape=(num_envs,), default 1.

    Returns
    -------
    RuntimeProvenanceV1
        Validated RuntimeProvenanceV1 with Python/library versions, default
        backend, first device, runtime version and current JAX precision setting.
        driver_version is None because this function does not query the driver.

    Raises
    ------
    RuntimeError
        The selected backend has no device or its precision flag
        is not a boolean. JAX initialization errors also propagate.
    ValueError
        Supplied metadata violates the provenance model.

    Host-only: imports JAX here, may initialize the default backend, and reads
    installed distribution metadata. Missing distributions propagate their
    lookup error. No device timing, hardware speed claim or file write occurs.
    """

    import jax

    backend = jax.default_backend()
    runtime_devices = cast(
        list[_RuntimeDevice],
        cast(object, jax.devices(backend)),
    )
    devices = tuple(runtime_devices)
    if not devices:
        raise RuntimeError("the selected JAX backend exposes no runtime device")
    device = devices[0]
    device_name = getattr(device, "device_kind", None) or str(device)
    platform_version = device.client.platform_version
    runtime_version = " ".join(platform_version.split()) or None
    x64_enabled = cast(bool, cast(object, jax.config.read("jax_enable_x64")))
    if type(x64_enabled) is not bool:
        raise RuntimeError("JAX x64 configuration did not return a boolean")

    return RuntimeProvenanceV1(
        python_version=host_platform.python_version(),
        package_version=package_version,
        jax_version=version("jax"),
        jaxlib_version=version("jaxlib"),
        numpy_version=version("numpy"),
        pydantic_version=version("pydantic"),
        platform=host_platform.system().lower() or sys.platform,
        machine=host_platform.machine() or "unknown",
        backend=backend,
        device=device_name,
        driver_version=None,
        runtime_version=runtime_version,
        precision="float64" if x64_enabled else "float32",
        environment_count=num_envs,
        batch_shape=(num_envs,),
        policy_execution_included=policy_execution_included,
    )


_DETERMINISTIC_FLAG = "--xla_gpu_deterministic_ops"


def read_declared_xla_flags() -> str:
    """Return the XLA_FLAGS string this process declared, or "" when unset.

    Returns
    -------
    str
        The environment variable exactly as set, spaces included.

    Notes
    -----
    Host-only; reads the environment, nothing else. This is what was declared,
    not proof of what XLA parsed: changing the string after the backend starts
    proves nothing. Training's execution identity and evaluation's
    execution_settings both read it here.
    """
    return os.environ.get("XLA_FLAGS", "")


def declared_gpu_deterministic_ops(flags: str) -> bool | None:
    """Say whether a declared XLA_FLAGS string turns deterministic GPU ops on.

    Parameters
    ----------
    flags : str
        An XLA_FLAGS string, split on spaces.

    Returns
    -------
    bool or None
        The last ``--xla_gpu_deterministic_ops`` setting wins: the bare flag
        or ``=true`` gives True, ``=false`` gives False, and any other value
        (``=1``, ``=True``) gives None, meaning unknown. No such flag gives
        False.
    """
    state: bool | None = False
    for token in flags.split():
        if token == _DETERMINISTIC_FLAG:
            state = True
        elif token.startswith(_DETERMINISTIC_FLAG + "="):
            setting = token.split("=", 1)[1]
            state = True if setting == "true" else False if setting == "false" else None
    return state


def capture_execution_settings() -> dict[str, object]:
    """Record the declared XLA settings of this attempt of a run.

    Returns
    -------
    dict
        ``declared_xla_flags_by_attempt``: a one-item list holding this
        attempt's whole XLA_FLAGS string; ``declared_gpu_deterministic_ops``:
        declared_gpu_deterministic_ops of that string.

    Notes
    -----
    Host-only. Evaluation and tournament passes save it beside
    runtime_provenance; a resumed pass merges attempts with
    merge_execution_settings. It records what was declared, not what XLA
    parsed: changing XLA_FLAGS after the backend starts proves nothing.
    """
    flags = read_declared_xla_flags()
    return {
        "declared_xla_flags_by_attempt": [flags],
        "declared_gpu_deterministic_ops": declared_gpu_deterministic_ops(flags),
    }


def merge_execution_settings(
    records: Sequence[Mapping[str, object] | None],
) -> dict[str, object]:
    """Combine execution_settings records, oldest first, into one record.

    Parameters
    ----------
    records : sequence of mapping or None
        One record per attempt or per pass. None stands for an attempt or
        pass that saved no record but has games.

    Returns
    -------
    dict
        ``declared_xla_flags_by_attempt``: every attempt's string in order,
        with None for an unrecorded one; ``declared_gpu_deterministic_ops``:
        False if any record says False, otherwise None if any is None or
        unrecorded, otherwise True. An empty sequence gives an empty list and
        None.
    """
    flags: list[object] = []
    states: list[object] = []
    for record in records:
        if record is None:
            flags.append(None)
            states.append(None)
            continue
        attempts = record.get("declared_xla_flags_by_attempt")
        flags.extend(
            list(cast(Sequence[object], attempts))
            if isinstance(attempts, list)
            else [None]
        )
        states.append(record.get("declared_gpu_deterministic_ops"))
    merged = (
        False
        if any(state is False for state in states)
        else None
        if not states or any(state is not True for state in states)
        else True
    )
    return {
        "declared_xla_flags_by_attempt": flags,
        "declared_gpu_deterministic_ops": merged,
    }


__all__ = [
    "capture_debugger_runtime_provenance_v1",
    "capture_execution_settings",
    "capture_runtime_provenance",
    "declared_gpu_deterministic_ops",
    "merge_execution_settings",
    "read_declared_xla_flags",
]
