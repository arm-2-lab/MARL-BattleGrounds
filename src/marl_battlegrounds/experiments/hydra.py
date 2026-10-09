"""Compose, preview and launch experiments through the existing public API.

Install marl-battlegrounds[hydra] to use this module. Hydra composes settings;
MARL-BGs owns validation, execution and saved records. On Python 3.14 a small
compatibility patch affects only Hydra parser creation, before worker startup.
It never edits installed dependencies or leaves argparse validation disabled.
"""

from __future__ import annotations

import argparse

# These imports remain host-only. Import training execution inside run_training.
import json
import math
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from functools import wraps
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from threading import current_thread, main_thread
from typing import TYPE_CHECKING, Any, cast

from marl_battlegrounds.training.config import (
    TrainConfig,
    config_from_dict,
    config_to_dict,
)
from marl_battlegrounds.training.study import (
    _config_paths,  # pyright: ignore[reportPrivateUsage]
    _path,  # pyright: ignore[reportPrivateUsage]
    _runtime_callbacks,  # pyright: ignore[reportPrivateUsage]
    _runtime_paths,  # pyright: ignore[reportPrivateUsage]
)

try:
    from omegaconf import DictConfig, OmegaConf

    import_module("hydra")
    import_module("omegaconf")
except ImportError as error:
    raise ImportError(
        "Hydra launchers need the optional extra. Install marl-battlegrounds[hydra]."
    ) from error

_parser_patched = False


def _compatible_args_parser(
    factory: Callable[[], argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Build Hydra 1.3.7's parser under Python 3.14's stricter help check.

    factory is the original Hydra parser constructor. Only its exact nested
    LazyCompletionHelp object for shell_completion bypasses _check_help; normal
    argparse actions retain their checks. Always restore the original method,
    including on failure. Call on the main thread before starting workers.
    No task, backend, files or command-line parsing runs here.
    """
    if current_thread() is not main_thread():
        raise RuntimeError("Create the Hydra parser on the main thread before workers")
    original = cast(
        Callable[[argparse.ArgumentParser, argparse.Action], None],
        getattr(argparse.ArgumentParser, "_check_help"),  # noqa: B009
    )

    def check(parser: argparse.ArgumentParser, action: argparse.Action) -> None:
        """Delegate ordinary help; allow only Hydra's known lazy completion help."""
        help_type = type(action.help)
        if (
            action.dest == "shell_completion"
            and help_type.__module__ == "hydra._internal.utils"
            and help_type.__qualname__ == "get_args_parser.<locals>.LazyCompletionHelp"
        ):
            return None
        return original(parser, action)

    setattr(argparse.ArgumentParser, "_check_help", check)  # noqa: B010
    try:
        return factory()
    finally:
        setattr(argparse.ArgumentParser, "_check_help", original)  # noqa: B010


def prepare_hydra() -> None:
    """Install the narrow parser backport once for Python 3.14 and Hydra 1.3.7.

    Other Python versions use the upstream parser unchanged. On Python 3.14 an
    unqualified Hydra version raises RuntimeError with the supported install
    command. Patch both upstream references, including hydra.main's imported
    name. Call on the main thread before workers. Repeated calls do nothing.
    This changes no installed file and starts no application or numerical work.
    """
    global _parser_patched
    if _parser_patched or sys.version_info[:2] != (3, 14):
        return
    if version("hydra-core") != "1.3.7":
        raise RuntimeError(
            "Python 3.14 Hydra launchers require the qualified hydra-core==1.3.7. "
            "Install marl-battlegrounds[hydra]."
        )
    if current_thread() is not main_thread():
        raise RuntimeError("Prepare Hydra on the main thread before starting workers")
    utils = import_module("hydra._internal.utils")
    main_module = import_module("hydra.main")
    original = cast(Callable[[], argparse.ArgumentParser], utils.get_args_parser)

    @wraps(original)
    def parser() -> argparse.ArgumentParser:
        """Apply the compatibility check only while creating Hydra's parser."""
        return _compatible_args_parser(original)

    setattr(utils, "get_args_parser", parser)  # noqa: B010
    setattr(main_module, "get_args_parser", parser)  # noqa: B010
    _parser_patched = True


if TYPE_CHECKING:
    from hydra.experimental.callback import Callback

    from marl_battlegrounds.evaluation.results import EvaluationResult
    from marl_battlegrounds.training.runner import TrainResult

_ADAPTER_KEYS = frozenset(
    {
        "output_dir",
        "resume_from",
        "preview",
        "runtime",
        "resources",
        "name",
        "recipe",
        "opponent_pool",
        "study_options",
    }
)


def _plain_settings(value: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one settings mapping and reject nonfinite or non-JSON values.

    Resolve interpolations and reject mandatory missing values. Mapping keys
    must be strings; leaves must be null, Boolean, integer, finite float or text.
    Return a fresh dictionary without changing value or opening declared inputs.
    OmegaConf errors retain the original setting path for useful CLI errors.
    """
    if not isinstance(cast(object, value), (DictConfig, Mapping)):
        raise TypeError("Settings must be a mapping")
    cfg = value if isinstance(value, DictConfig) else OmegaConf.create(dict(value))
    plain = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    def check(item: object, location: str) -> None:
        """Check one resolved subtree, naming the first unsupported value."""
        if isinstance(item, dict):
            for key, child in cast(dict[object, object], item).items():
                if not isinstance(key, str):
                    raise ValueError(f"{location} needs string mapping keys")
                check(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(cast(list[object], item)):
                check(child, f"{location}[{index}]")
        elif type(item) is float and not math.isfinite(item):
            raise ValueError(f"{location} must be finite")
        elif item is not None and type(item) not in (str, bool, int, float):
            raise TypeError(f"{location} must contain plain JSON values")

    if not isinstance(plain, dict):
        raise TypeError("Settings must be a mapping")
    check(cast(dict[object, object], plain), "Settings")
    return cast(dict[str, Any], plain)


def resolve_training_config(
    value: DictConfig | Mapping[str, Any], *, base_dir: str | Path
) -> TrainConfig:
    """Convert composed settings through the ordinary training config parser.

    Parameters
    ----------
    value : DictConfig or mapping
        Training fields plus optional launcher controls. Interpolations are
        resolved before parsing. A named opponent_pool replaces the whole
        opponents declaration; supplying both non-null values is an error.
        Only documented launcher keys are removed. Wrong-method learner blocks
        and unknown fields remain errors from the shared API parser.
    base_dir : str or Path
        Original launch directory for relative input paths, including paths
        with spaces. Registered names and module:function references stay names.

    Returns
    -------
    TrainConfig
        Validated settings with API defaults and method-specific cadence.

    Raises
    ------
    ValueError, TypeError
        Settings are not a finite plain mapping, conflict, or fail API checks.
    omegaconf.errors.OmegaConfBaseException
        An interpolation cannot resolve or a mandatory value is missing.

    Notes
    -----
    Starts no numerical library, model, factory or worker. It does not verify
    files, payloads, resources or external services. Live Python inputs still
    belong to train's keyword arguments, outside serialized configuration.
    """
    plain = _plain_settings(value)
    pool = plain.get("opponent_pool")
    if pool is not None:
        if plain.get("opponents") is not None:
            raise ValueError("Choose either a named pool or inline opponents, not both")
        plain["opponents"] = pool
    settings = {key: item for key, item in plain.items() if key not in _ADAPTER_KEYS}
    base = Path(base_dir).expanduser().resolve()
    runtime = _runtime_paths(plain.get("runtime", {}), base)
    return config_from_dict(_config_paths(settings, base, runtime=runtime))


def _launch_directory(
    output_dir: object,
    name: object,
    resume_from: object = None,
    fallback_output_dir: object = None,
) -> str:
    """Choose a unique Hydra record folder without creating it.

    output_dir is the exact optional trainer destination. resume_from, when
    present, names a run or learner checkpoint whose run owns the new records.
    fallback_output_dir supplies a second lesson destination only when
    output_dir is None. Its records are kept
    in a sibling <name>.launches folder. Otherwise use artifacts/experiments/name.
    name must be one nonempty path component. IDs use UTC microseconds and a
    random suffix. Fresh paths need no input reads; recovery reads and checks
    small checkpoint metadata, without loading model arrays.
    """
    if (
        not isinstance(name, str)
        or not name.strip()
        or name in (".", "..")
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("name must be one nonempty folder name")
    launch_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    )
    if resume_from is not None:
        from marl_battlegrounds.training._checkpoint_metadata import (
            read_resume_checkpoint,
        )

        checkpoint, _ = read_resume_checkpoint(cast(str, resume_from))
        output_dir = str(checkpoint.parent.parent)
    if output_dir is None:
        output_dir = fallback_output_dir
    if output_dir is None:
        parent = Path.cwd() / "artifacts" / "experiments" / name
    else:
        destination = Path(_path(output_dir, Path.cwd(), "output_dir"))
        parent = destination.with_name(destination.name + ".launches")
    return str(parent / launch_id)


OmegaConf.register_new_resolver("marl_launch_dir", _launch_directory, use_cache=True)


def run_training(
    value: DictConfig | Mapping[str, Any],
    *,
    base_dir: str | Path,
    launch_dir: str | Path,
) -> TrainResult | dict[str, Any]:
    """Preview or run one composed training job through the public API.

    Parameters
    ----------
    value : DictConfig or mapping
        Fresh settings, or sparse settings with resume_from naming a saved run
        or learner checkpoint. preview=true returns a non-executing report.
        runtime supplies reloadable bindings through the ordinary train keywords.
        Explicit resources select one device before numerical imports. Sweeps
        use the shared study owner.
    base_dir : str or Path
        Original launch directory used for relative inputs and output_dir.
    launch_dir : str or Path
        Hydra's record directory. With no output_dir the trainer receives its
        run child. A supplied output_dir is passed as the exact destination.

    Returns
    -------
    TrainResult or dict[str, Any]
        The API result on success, or a finite preview record with effective
        settings, budget, input references, output paths and unperformed checks.
        Prints the record or actual returned run path for command-line callers.

    Notes
    -----
    Preview creates no files itself. Hydra may already have created its records.
    Execution saves effective settings beside Hydra records and lets train own
    all run files. Reusing a nonempty direct-call launch directory is rejected.
    Hydra may supply its own new records, but effective settings never overwrite
    an earlier declaration. Errors print output locations before propagating.
    """
    from hydra.core.hydra_config import HydraConfig

    plain = _plain_settings(value)
    if "resume_from" in plain and plain["resume_from"] is None:
        raise ValueError("resume_from must name a saved run or learner checkpoint")
    if type(plain.get("preview", False)) is not bool:
        raise TypeError("preview must be a Boolean")
    if HydraConfig.initialized():
        chosen = HydraConfig.get().runtime.choices.get("method")
        if chosen is not None and plain.get("method") != chosen:
            raise ValueError("method must agree with the selected method group")
    runtime = _runtime_paths(
        plain.get("runtime", {}), Path(base_dir).expanduser().resolve()
    )
    checkpoint = None
    supplied_config = None
    if plain.get("resume_from") is not None:
        checkpoint, config, supplied_config = _resume_settings(plain, Path(base_dir))
    else:
        config = resolve_training_config(plain, base_dir=base_dir)
        supplied_config = config
    launch = Path(launch_dir).expanduser().resolve()
    destination = (
        launch / "run"
        if plain.get("output_dir") is None
        else Path(_path(plain["output_dir"], Path(base_dir), "output_dir"))
    )
    if checkpoint is not None:
        destination = checkpoint.parent.parent
    from marl_battlegrounds.evaluation.revision import (
        _check_output_path,  # pyright: ignore[reportPrivateUsage]
    )

    _check_output_path(destination)
    _check_output_path(launch)
    from marl_battlegrounds.training import _launch

    resources = _launch.resource_settings(
        plain.get("resources"), base_dir=Path(base_dir)
    )
    if resources is not None and (
        len(resources["devices"]) != 1 or resources["devices"][0]["max_workers"] != 1
    ):
        raise ValueError(
            "A single run needs one device and max_workers=1; "
            "use -m or a study for more workers"
        )
    environment = _launch.resource_environment(
        None if resources is None else resources["devices"][0],
        compilation_cache=None
        if resources is None
        else resources.get("compilation_cache"),
    )
    report = {
        "mode": "Preview: no training or workers started",
        "case_count": 1,
        "total_env_steps": config.total_env_steps,
        "config": config_to_dict(config),
        "runtime": runtime,
        "resources": resources,
        "resource_environment": _launch.resource_facts(environment),
        "output_dir": str(destination),
        "launch_dir": str(launch),
        "resume_from": None if checkpoint is None else str(checkpoint),
        "pending_checks": [
            "Input payloads and factories",
            "Numerical setup and resources",
            "Training output availability",
        ],
    }
    if plain.get("study_options"):
        raise ValueError("study_options requires a study or -m sweep")
    if plain.get("preview", False):
        print(json.dumps(report, indent=2, allow_nan=False))
        return report
    if resources is not None:
        if "jax" in sys.modules:
            raise RuntimeError(
                "Apply resources in a fresh launcher process before importing JAX"
            )
        _launch.check_resource_devices(resources)
        os.environ.update(environment)
    if HydraConfig.initialized():
        from hydra.types import RunMode

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            raise ValueError(
                "Training sweeps require the study launcher; no case was started"
            )
    from marl_battlegrounds.training import train
    from marl_battlegrounds.training._run_io import atomic_json

    records_subdir = (
        HydraConfig.get().output_subdir if HydraConfig.initialized() else ".hydra"
    )
    records = (launch / (records_subdir or ".hydra")).resolve()
    hydra_owns_launch = (
        HydraConfig.initialized()
        and Path(HydraConfig.get().runtime.output_dir).resolve() == launch
    )
    if (
        not hydra_owns_launch
        and launch.exists()
        and (not launch.is_dir() or any(launch.iterdir()))
    ):
        raise ValueError(f"Launch record directory must be new or empty: {launch}")
    if records != launch and launch not in records.parents:
        raise ValueError("Hydra records must stay inside this launch directory")
    if (records / "training_config.json").exists():
        raise ValueError(f"Launch already has effective training settings: {records}")
    records.mkdir(parents=True, exist_ok=True)
    atomic_json(records / "training_config.json", config_to_dict(config))
    atomic_json(
        records / "resources.json",
        {
            "requested": resources,
            "environment": _launch.resource_facts(os.environ),
        },
    )
    try:
        with ExitStack() as stack:
            runtime = _runtime_callbacks(runtime, stack)
            result = (
                train(supplied_config, output_dir=destination, **runtime)
                if checkpoint is None
                else train(supplied_config, resume_from=checkpoint, **runtime)
            )
    except Exception:
        print(
            f"Training output: {destination}\nLaunch records: {launch}", file=sys.stderr
        )
        raise
    print(f"Training output: {result.run_dir}\nLaunch records: {launch}")
    return result


prepare_hydra()


def _study_sweep(  # pyright: ignore[reportUnusedFunction]
    jobs: list[dict[str, Any]],
    *,
    overrides: list[list[str]],
    base_dir: Path,
    launch_dir: Path,
    sweep_overrides: list[str],
    sweeper_params: object,
) -> dict[str, Any]:
    """Convert one complete Hydra batch and hand it to the existing study owner.

    jobs are fully resolved task mappings in Hydra order. overrides retains each
    job's exact override list; sweep_overrides/params retain the original grid.
    base_dir is the original working directory, launch_dir holds Hydra records.
    Check every case and shared control before workers or study output exist.
    Preview uses the same cases and paths without importing models/factories.
    Input identities and device access remain execution checks. Return the study
    report with its saved location, or explicitly labelled preview case records.
    """
    from marl_battlegrounds.evaluation.revision import (
        _check_output_path,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training import _launch, study
    from marl_battlegrounds.training._run_io import atomic_json

    if not jobs or len(jobs) != len(overrides):
        raise ValueError("A sweep needs one override list per composed job")
    launch = launch_dir.expanduser().resolve()
    _check_output_path(launch)
    cases: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    shared: dict[str, Any] | None = None
    duplicates: set[str] = set()
    artifact_cache: dict[str, dict[str, Any]] = {}
    for index, plain in enumerate(jobs):
        if "resume_from" in plain:
            raise ValueError("Resume the saved study; do not re-expand a Hydra sweep")
        config = config_to_dict(resolve_training_config(plain, base_dir=base_dir))
        runtime = _runtime_paths(plain.get("runtime", {}), base_dir)
        options = study._object(plain.get("study_options", {}), "study_options")  # pyright: ignore[reportPrivateUsage]
        study._keys(options, {"failure_policy", "duration_seconds"}, "study_options")  # pyright: ignore[reportPrivateUsage]
        failure = options.get("failure_policy", "stop")
        if failure not in ("stop", "continue"):
            raise ValueError("study_options.failure_policy must be stop or continue")
        duration = study._duration(  # pyright: ignore[reportPrivateUsage]
            options.get("duration_seconds"), "study duration_seconds"
        )
        destination = (
            launch / "study"
            if plain.get("output_dir") is None
            else Path(_path(plain["output_dir"], base_dir, "output_dir"))
        )
        _check_output_path(destination)
        if destination == launch or destination in launch.parents:
            raise ValueError(
                "Hydra records must sit outside the experiment destination"
            )
        preview = plain.get("preview", False)
        if type(preview) is not bool:
            raise ValueError("preview must be true or false")
        name = plain.get("name", "training")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Study name must be nonempty")
        controls = {
            "output_dir": str(destination),
            "name": name,
            "preview": preview,
            "resources": _launch.resource_settings(
                plain.get("resources"), base_dir=base_dir
            ),
            "failure_policy": failure,
            "duration_seconds": duration,
        }
        if shared is not None and shared != controls:
            raise ValueError(
                "All sweep jobs must share output_dir, name, preview, "
                "resources and study_options"
            )
        shared = controls
        seed = config.pop("seed")
        identifier = f"{config['method']}-job-{index:04d}"
        case = {"id": identifier, "config": config, "seeds": [seed], "runtime": runtime}
        identity = study._effective_job_identity(  # pyright: ignore[reportPrivateUsage]
            case, seed=seed, artifact_cache=artifact_cache
        )
        if identity in duplicates:
            raise ValueError(f"Duplicate effective Hydra job at index {index}")
        duplicates.add(identity)
        cases.append(case)
        full_id = f"{identifier}--seed-{seed}"
        mapping.append(
            {
                "id": full_id,
                "overrides": overrides[index],
                "run_dir": str(destination / "cases" / full_id / "run"),
            }
        )
    assert shared is not None
    declaration = {
        "name": shared["name"],
        "failure_policy": shared["failure_policy"],
        "duration_seconds": shared["duration_seconds"],
        "cases": cases,
        "launch": {
            "kind": "hydra",
            "directory": str(launch),
            "overrides": sweep_overrides,
            "sweeper_params": sweeper_params,
            "jobs": mapping,
        },
    }
    if shared["resources"] is not None:
        declaration["resources"] = shared["resources"]
    report = {
        "preview": shared["preview"],
        "case_count": len(cases),
        "output_dir": shared["output_dir"],
        "launch_dir": str(launch),
        "total_env_steps": sum(case["config"]["total_env_steps"] for case in cases),
        "cases": [{**row, "status": "preview"} for row in mapping],
        "declaration": declaration,
        "resource_environment": [
            _launch.resource_facts(
                _launch.resource_environment(
                    None if slot is None else slot["device"],
                    compilation_cache=None
                    if slot is None
                    else slot.get("compilation_cache"),
                )
            )
            for slot in study._resource_slots(declaration)  # pyright: ignore[reportPrivateUsage]
        ],
        "pending_checks": [
            "Input and factory identities, including aliased duplicate jobs",
            "Input payloads",
            "Numerical setup and available devices",
        ],
    }
    if shared["preview"]:
        print(json.dumps(report, indent=2, allow_nan=False))
        return report
    records = launch / ".hydra"
    records.mkdir(parents=True, exist_ok=True)
    _launch.write_once(records / "ordered_jobs.json", {"jobs": jobs})
    _launch.write_once(records / "study_declaration.json", declaration)
    result = study.run_study(declaration, output_dir=shared["output_dir"])
    result = {**result, "report_path": str(Path(shared["output_dir"]) / "report.json")}
    atomic_json(records / "study_result.json", result)
    print(
        json.dumps(
            {
                "study_dir": shared["output_dir"],
                "report": result["report_path"],
                "launch_dir": str(launch),
            },
            indent=2,
        )
    )
    return result


def _actor_description(reference: str | Path) -> dict[str, Any]:
    """Read a saved actor's checked description from any saved-actor form.

    reference is an actor export or learner folder (read_checkpoint_description,
    exactly as before), a portable model file (its embedded original
    ``actor_details.json``, after every part's SHA-256 is checked) or an
    ``hf://`` model address (the released file, downloaded once into Hugging
    Face's standard cache when missing and checked against its repo's
    ``steps.csv``). Returns the description dict. Raises ValueError when a
    check fails or the reference is none of these.
    """
    from marl_battlegrounds.training._checkpoint_metadata import (
        read_checkpoint_description,
    )

    text = str(reference)
    if text.startswith("hf://"):
        from marl_battlegrounds.released_models import (
            _fetch,  # pyright: ignore[reportPrivateUsage]
        )

        reference = cast(Path, _fetch(text).path)
    if Path(reference).is_file():
        from marl_battlegrounds.portable_models import read_portable_model

        return read_portable_model(reference).actor_details
    return read_checkpoint_description(reference)


def _evaluation_paths(
    checkpoint: object,
    opponent: object,
    *,
    base_dir: Path,
    launch_dir: str | Path,
    output_dir: object = None,
    resume_from: object = None,
    compilation_cache: object = None,
) -> tuple[Path | str, str]:
    """Check evaluation destinations before any launcher or evaluator writes.

    checkpoint names an existing artifact directory, a portable model file or
    a released model (a typed name such as
    ``MARL-BattleGrounds/tdm-season-0-qmix@final``, an ``hf://`` address or a
    released-layout folder); opponent uses the shared reference grammar.
    Relative paths use base_dir. launch_dir and its .hydra receipts, output_dir
    and optional compilation_cache must be outside local input artifacts.
    Launcher/cache records also stay outside a resumed result folder. Reject
    links before resolving parent components. Return the absolute actor path
    (or a released model's saved form, its exact ``hf://`` address or chosen
    file path) and the normalized opponent reference, released names included.
    No payloads or factories are read; a typed name may read its repo's small
    index files.
    """
    from marl_battlegrounds._method_loading import (
        _reference_kind,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.evaluation.revision import (
        _check_output_path,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training._checkpoint_metadata import (
        _directory,  # pyright: ignore[reportPrivateUsage]
    )

    def path(value: object, label: str) -> Path:
        """Resolve a named path after rejecting links along its lexical parents."""
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise TypeError(f"{label} must be a nonempty path")
        target = Path(value).expanduser()
        target = (target if target.is_absolute() else base_dir / target).absolute()
        if any(part.is_symlink() for part in (*target.parents, target)):
            raise ValueError(f"{label} must not follow symbolic links")
        return target.resolve()

    from marl_battlegrounds.released_models import (
        _saved_reference,  # pyright: ignore[reportPrivateUsage]
    )

    inputs: list[Path] = []
    actor: Path | str
    if (
        isinstance(checkpoint, str)
        and _reference_kind(checkpoint, base_dir=base_dir) == "released"
    ):
        actor = _saved_reference(checkpoint, base_dir=base_dir)
    else:
        actor = path(checkpoint, "checkpoint")
        actor = actor if actor.is_file() else _directory(actor)
        inputs.append(actor)
    if not isinstance(opponent, str):
        raise TypeError("opponent must be a method reference string")
    kind = _reference_kind(opponent, base_dir=base_dir)
    if kind == "artifact":
        other = path(opponent, "opponent")
        other = other if other.is_file() else _directory(other)
        inputs.append(other)
        opponent = str(other)
    elif kind == "released":
        opponent = _saved_reference(opponent, base_dir=base_dir)
    elif kind == "unknown":
        raise ValueError(f"Unknown opponent reference: {opponent}")
    launch = path(launch_dir, "launch_dir")
    writes = [launch, path(launch / ".hydra", "Evaluation receipts")]
    if compilation_cache is not None:
        writes.append(path(compilation_cache, "compilation_cache"))
    results = [
        path(value, name)
        for name, value in (("output_dir", output_dir), ("resume_from", resume_from))
        if value is not None
    ]
    for target in writes:
        if any(target == root or root in target.parents for root in results):
            raise ValueError("Launch and cache records must stay outside results")
    if output_dir is not None:
        writes.append(path(output_dir, "output_dir"))
    for target in writes:
        _check_output_path(target)
        if any(target == root or root in target.parents for root in inputs):
            raise ValueError("Evaluation outputs must stay outside input artifacts")
    return actor, opponent


def _preparation_paths(
    opponent_bundle: str | Path,
    output_dir: str | Path,
    *,
    launch_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Check preparation destinations before records or numerical setup.

    The bundle owns every declared actor below it. Keep protocol and Hydra writes
    outside that input tree. Reject symbolic links in all paths before resolving
    them. Return absolute bundle and output roots without creating either one.
    A common parent for inputs and outputs is allowed; overlapping writes are not.
    """
    from marl_battlegrounds.evaluation.revision import (
        _check_output_path,  # pyright: ignore[reportPrivateUsage]
    )

    def checked(value: str | Path) -> Path:
        """Keep lexical parents long enough to reject symbolic path aliases."""
        target = Path(value).expanduser().absolute()
        if any(part.is_symlink() for part in (*target.parents, target)):
            raise ValueError("Preparation paths must not follow symbolic links")
        return target.resolve()

    bundle, root = checked(opponent_bundle), checked(output_dir)
    if root == bundle or bundle in root.parents:
        raise ValueError("Preparation outputs must stay outside the opponent bundle")
    writes = [checked(root / "protocol")]
    if launch_dir is not None:
        writes.append(checked(launch_dir))
    for target in writes:
        if target == bundle or bundle in target.parents or target in bundle.parents:
            raise ValueError(
                "Preparation outputs must stay outside the opponent bundle"
            )
        _check_output_path(target)
    return bundle, root


def _prepare_launch(  # pyright: ignore[reportUnusedFunction]
    mode: object,
    output_dir: object,
    resume_from: object,
    launch_dir: str,
    records_subdir: str | None,
    require_resume: bool = False,
    sweep_dir: str | None = None,
    launcher_target: str | None = None,
    additional_output_dir: object = None,
    evaluation_inputs: Mapping[str, Any] | None = None,
    opponent_bundle: str | None = None,
    scenario_inputs: Mapping[str, Any] | None = None,
) -> Callback:
    """Check paths and launcher ownership before Hydra creates any outputs.

    Hydra constructs this callback before job dispatch. mode is its RunMode;
    output_dir/resume_from are root controls. launch_dir and records_subdir are
    actual Hydra destinations. sweep_dir is the multirun record folder and
    launcher_target must name the study plugin for multiruns. Both default to
    None for existing single-run callers. require_resume marks the sparse resume preset;
    it requires a recovery path even if an override clears or removes that field.
    Reject unsupported sweeps and source-changing
    output paths before writes. additional_output_dir protects a lesson
    destination with a different name, such as fixture directory. Return Hydra's
    ordinary no-op callback. evaluation_inputs supplies checkpoint, opponent and
    resources for the evaluation preflight; None leaves other launchers unchanged. Reads
    small recovery records and Git ignore/index rules only, with no backend.
    opponent_bundle protects preparation inputs from protocol and Hydra writes.
    scenario_inputs supplies system, opponent and resources for exact-scenario
    lessons; check method/result paths before launch records or caches are made.
    """
    from hydra.experimental.callback import Callback
    from hydra.types import RunMode

    from marl_battlegrounds.evaluation.revision import (
        _check_output_path,  # pyright: ignore[reportPrivateUsage]
    )

    if mode == RunMode.MULTIRUN:
        if (
            launcher_target != "hydra_plugins.marl_battlegrounds_study.StudyLauncher"
            or require_resume
        ):
            raise ValueError(
                "Sweep preview and execution require the study launcher; "
                "resume the saved study separately"
            )
        if sweep_dir is None:
            raise ValueError("Study sweeps require a launch-record directory")
        launch_dir = sweep_dir
    if require_resume and (not isinstance(resume_from, str) or not resume_from.strip()):
        raise ValueError("resume_from must name a saved run or learner checkpoint")
    if evaluation_inputs is not None:
        resources = evaluation_inputs.get("resources")
        cache = None if resources is None else resources.get("compilation_cache")
        _evaluation_paths(
            evaluation_inputs.get("checkpoint"),
            evaluation_inputs.get("opponent"),
            base_dir=Path.cwd(),
            launch_dir=launch_dir,
            output_dir=output_dir,
            resume_from=resume_from,
            compilation_cache=cache,
        )
    if scenario_inputs is not None:
        from marl_battlegrounds._method_loading import (
            _method_output_paths,  # pyright: ignore[reportPrivateUsage]
        )

        resources = scenario_inputs.get("resources")
        cache = None if resources is None else resources.get("compilation_cache")
        extras = [Path(launch_dir)]
        if records_subdir is not None:
            extras.append(Path(launch_dir) / records_subdir)
        if cache is not None:
            extras.append(Path(cache))
        _method_output_paths(
            [scenario_inputs.get("system"), scenario_inputs.get("opponent")],
            output_dir=cast(str | None, output_dir),
            resume_from=cast(str | None, resume_from),
            additional_outputs=extras,
            result_children=[
                f"scenario-{identifier}"
                for identifier in scenario_inputs.get("scenario_ids", ())
            ],
        )
    if opponent_bundle is not None:
        if not isinstance(output_dir, (str, Path)):
            raise ValueError("Preparation requires an output_dir path")
        _preparation_paths(opponent_bundle, output_dir, launch_dir=launch_dir)
    launch = Path(launch_dir).expanduser().resolve()
    _check_output_path(launch)
    if launch.exists() and (not launch.is_dir() or any(launch.iterdir())):
        raise ValueError(f"Launch record directory must be new or empty: {launch}")
    for output in (output_dir, additional_output_dir):
        if output is None:
            continue
        destination = Path(_path(output, Path.cwd(), "output_dir"))
        _check_output_path(destination)
        if destination == launch or destination in launch.parents:
            raise ValueError(
                "Hydra records must sit outside the experiment destination"
            )
    if records_subdir is not None:
        records = (launch / records_subdir).resolve()
        if records != launch and launch not in records.parents:
            raise ValueError(
                "hydra.output_subdir must stay inside this launch directory"
            )
        _check_output_path(records)
    if resume_from is not None:
        if output_dir is not None:
            raise ValueError("Resume inherits its run directory; omit output_dir")
    elif output_dir is not None:
        _check_output_path(Path(_path(output_dir, Path.cwd(), "output_dir")))
    else:
        _check_output_path(launch / "run")
    return Callback()


def _resume_settings(
    plain: dict[str, Any], base_dir: Path
) -> tuple[Path, TrainConfig, TrainConfig | None]:
    """Read and compare sparse recovery assertions without loading model arrays.

    plain holds only explicitly supplied scientific fields plus launcher controls.
    Inherit saved defaults through the runner's continuation-aware reader. Merge
    only learner setting leaves; replace complete population/list declarations.
    Omitted scientific settings return None for train's config argument. Explicit
    null or a differing effective setting raises before recovery writes. Released
    model names resolve against the commits the saved run recorded. Returns
    checkpoint path, inherited/equal effective config and optional asserted config.
    """
    from marl_battlegrounds.baselines.methods import method_settings_field
    from marl_battlegrounds.training._checkpoint_metadata import (
        _require_current_schemas,  # pyright: ignore[reportPrivateUsage]
        read_resume_checkpoint,
    )
    from marl_battlegrounds.training.runner import (
        _child_config,  # pyright: ignore[reportPrivateUsage]
        _saved_config,  # pyright: ignore[reportPrivateUsage]
    )

    if plain.get("output_dir") is not None:
        raise ValueError("Resume inherits its run directory; omit output_dir")
    checkpoint, description = read_resume_checkpoint(
        plain["resume_from"], base_dir=base_dir
    )
    _require_current_schemas(description)
    inherited = _saved_config(description)
    overlay = {key: item for key, item in plain.items() if key not in _ADAPTER_KEYS}
    if plain.get("opponent_pool") is not None:
        if overlay.get("opponents") is not None:
            raise ValueError("Choose either a named pool or inline opponents, not both")
        overlay["opponents"] = plain["opponent_pool"]
    if not overlay:
        return checkpoint, inherited, None
    from marl_battlegrounds.released_models import (
        _recorded_commits,  # pyright: ignore[reportPrivateUsage]
    )

    saved = config_to_dict(inherited)
    base = base_dir.expanduser().resolve()
    # Released names resolve to the commits the saved run recorded.
    supplied = _config_paths(
        overlay,
        base,
        runtime=_runtime_paths(plain.get("runtime", {}), base),
        commits=_recorded_commits(description),
    )
    selected = method_settings_field(inherited.method)
    for key, item in supplied.items():
        if key == selected and isinstance(item, dict):
            saved[key] = {**saved[key], **item}
        else:
            saved[key] = item
    continuation = description["metadata"].get("continuation")
    if continuation is None:
        compared = config_from_dict(saved)
    else:
        compared = _child_config(
            saved,
            total_env_steps=saved["total_env_steps"],
            root_total_env_steps=continuation["segment"]["root_schedule"][
                "total_env_steps"
            ],
        )
    if config_to_dict(compared) != config_to_dict(inherited):
        raise ValueError("Resume config differs from the saved experiment")
    return checkpoint, compared, compared


def example_main[T](
    name: str, run: Callable[..., T], *, report: Callable[[T], None] | None = None
) -> int:
    """Compose one tutorial's settings and call its ordinary Python lesson.

    Parameters
    ----------
    name : str
        Packaged examples/<name>.yaml root, matching the public example file.
    run : callable
        The example's plain keyword-callable workflow. Its signature rejects
        unknown settings. Import numerical work inside that workflow so help,
        config display and preview remain host-only. A returned integer becomes
        the process exit status; other returns mean success.
    report : callable or None, default None
        Optional display function for the workflow's returned value. Without it,
        mappings print as JSON; other non-integer values are not printed.

    Returns
    -------
    int
        Zero on success, or the lesson's integer status. API exceptions propagate
        after printing the saved launch directory. Hydra may exit for CLI errors.

    Notes
    -----
    Settings resolve through the same finite-value check as training. Resources
    apply before numerical imports. This helper starts no scheduler: complete
    tutorial sweeps are refused; examples/train.py owns training sweeps through
    the study launcher. Hydra records sit beside explicit outputs in a unique
    .launches folder. No optional game recording is enabled by this helper.
    Tutorial presets have their own search root. A plain setting such as
    method=ippo must not select a training configuration group instead.
    """
    import inspect

    import hydra
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd

    from marl_battlegrounds.training import _launch

    exit_code = 0

    @hydra.main(
        version_base="1.3",
        config_path="pkg://marl_battlegrounds.experiments.configs.examples",
        config_name=name,
    )
    def configured(config: DictConfig) -> None:
        """Validate resolved lesson inputs before entering its public API workflow."""
        nonlocal exit_code
        settings = _plain_settings(config)
        preview = settings.pop("preview", False)
        if type(preview) is not bool:
            raise TypeError("preview must be true or false")
        resources = _launch.resource_settings(
            settings.pop("resources", None), base_dir=Path(get_original_cwd())
        )
        if resources is not None and (
            len(resources["devices"]) != 1
            or resources["devices"][0]["max_workers"] != 1
        ):
            raise ValueError("A tutorial needs one device with max_workers=1")
        environment = _launch.resource_environment(
            None if resources is None else resources["devices"][0],
            compilation_cache=None
            if resources is None
            else resources.get("compilation_cache"),
        )
        inspect.signature(run).bind(**settings)
        location = Path(HydraConfig.get().runtime.output_dir)
        if preview:
            print(
                json.dumps(
                    {
                        "preview": True,
                        "example": name,
                        "settings": settings,
                        "launch_dir": str(location),
                        "resource_environment": _launch.resource_facts(environment),
                        "pending_checks": [
                            "Input identities",
                            "API settings",
                            "Numerical execution",
                        ],
                    },
                    indent=2,
                    allow_nan=False,
                )
            )
            return
        if resources is not None:
            if "jax" in sys.modules:
                raise RuntimeError("Set resources before importing numerical modules")
            _launch.check_resource_devices(resources)
            os.environ.update(environment)
        # Keep all plain relative paths on the declared launch directory, even
        # when a user overrides Hydra's default chdir=false.
        previous = Path.cwd()
        try:
            os.chdir(get_original_cwd())
            result = run(**settings)
            if report is not None:
                report(result)
            if type(result) is int:
                exit_code = result
            elif report is None and isinstance(result, Mapping):
                print(
                    json.dumps(
                        dict(cast(Mapping[str, Any], result)),
                        indent=2,
                        allow_nan=False,
                        default=str,
                    )
                )
        except BaseException:
            print(f"Tutorial launch records: {location}", file=sys.stderr)
            raise
        finally:
            os.chdir(previous)
        print(f"Tutorial launch records: {location}")

    prepare_hydra()
    configured()
    return exit_code


# Keep writer and live authored objects on the ordinary Python evaluator route.
_EVALUATION_FIELDS = frozenset(
    {
        "num_episodes",
        "maps",
        "spawn_mode",
        "system_roster",
        "opponent_roster",
        "seed",
        "num_envs",
        "keep_batch_size",
        "metrics",
        "full_metrics_episodes",
        "replay_episodes",
        "save_replays",
        "output_dir",
        "resume_from",
        "score_threshold",
        "max_steps",
        "red_zone_depth",
        "phase",
        "pass_id",
        "chunk_size",
    }
)


def run_evaluation(
    value: DictConfig | Mapping[str, Any],
    *,
    expected_method: str,
    base_dir: str | Path,
    launch_dir: str | Path,
) -> EvaluationResult | dict[str, Any]:
    """Evaluate an explicit saved actor through the existing public evaluator.

    Parameters
    ----------
    value : DictConfig or mapping
        checkpoint is an actor or complete learner directory, a portable model
        file or a released model (a typed name such as
        ``MARL-BattleGrounds/tdm-season-0-qmix@final``, an ``hf://`` address or
        a released-layout folder). opponent is a built-in name, saved artifact
        or model file path, released model or trusted module:function factory.
        Released names are recorded as their exact ``hf://`` addresses.
        evaluation holds serializable evaluate keywords; writer and live objects
        belong to Python evaluate calls. Sparse resume omits scientific settings
        to inherit them; supplied values remain assertions. Missing resume counts
        are read from the exact saved phase/pass, never completed-row counts.
        preview=True reads descriptions only and returns the pending checks.
    expected_method : str
        Fixed built-in learner identity for this entry point. Saved schemas must
        match; filenames and selected/final labels are not method authorities.
    base_dir : path
        Original working directory for relative artifact and result paths.
    launch_dir : path
        Fresh launcher-record directory outside the input artifacts and results.
        Hydra may already have written its own settings here. This call adds one
        request and result receipt; an existing request is never overwritten.

    Returns
    -------
    EvaluationResult or dict[str, Any]
        The unchanged public API result, or a host-only preview mapping. A resumed
        result contains only newly played episodes; use load_results for all
        durable rows. result.run_dir names the actual saved run, or None when no
        game output was requested. Prints both result and launch locations.

    Raises
    ------
    ValueError, TypeError, OSError
        Settings, paths, descriptions or method identity are invalid. Factories,
        loaders and evaluation retain their own exception types and path notes.
        Wrong-method descriptions fail before model loading or result writes.

    Notes
    -----
    Applies explicit process resources before numerical imports. Loads each
    distinct reference once during setup. The evaluator owns snapshots, memory,
    information rights, recovery, metrics, replay and writing. No training,
    selection, hidden retry or alternate evaluator runs here. A released model
    is downloaded once into Hugging Face's standard cache when missing, also
    in preview, because its description is inside the model file. Otherwise
    preview does not verify payloads, import factories or initialize a
    numerical backend.
    Hydra itself can create launch records before this function is called.
    """
    from marl_battlegrounds._method_loading import (
        _method_loading_scope,  # pyright: ignore[reportPrivateUsage]
        _reference_kind,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.baselines.methods import validate_training_method
    from marl_battlegrounds.training import _launch
    from marl_battlegrounds.training._checkpoint_metadata import (
        _artifact_identity,  # pyright: ignore[reportPrivateUsage]
        _schema_method,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training._run_io import atomic_json
    from marl_battlegrounds.training.config import (
        _validate_red_zone_depth,  # pyright: ignore[reportPrivateUsage]
    )

    expected_method = validate_training_method(expected_method)
    plain = _plain_settings(value)
    unknown = set(plain) - {
        "checkpoint",
        "opponent",
        "evaluation",
        "preview",
        "resources",
        "name",
    }
    if unknown:
        raise ValueError(f"Unknown evaluation launcher settings: {sorted(unknown)}")
    if type(plain.get("preview", False)) is not bool:
        raise TypeError("preview must be true or false")
    settings = plain.get("evaluation", {})
    if not isinstance(settings, dict):
        raise TypeError("evaluation must be a mapping of evaluator settings")
    settings = cast(dict[str, Any], settings)
    unknown = set(settings) - _EVALUATION_FIELDS
    if unknown:
        raise ValueError(f"Unknown evaluation settings: {sorted(unknown)}")
    base = Path(base_dir).expanduser().absolute()

    def path(value: object, name: str) -> Path:
        """Keep lexical path components until the existing owner checks links."""
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{name} must be a nonempty path string")
        result = Path(value).expanduser()
        return (result if result.is_absolute() else base / result).absolute()

    from marl_battlegrounds.released_models import (
        _saved_reference,  # pyright: ignore[reportPrivateUsage]
    )

    raw_checkpoint = plain.get("checkpoint")
    checkpoint: Path | str = (
        _saved_reference(raw_checkpoint, base_dir=base)
        if isinstance(raw_checkpoint, str)
        and raw_checkpoint.strip()
        and _reference_kind(raw_checkpoint, base_dir=base) == "released"
        else path(raw_checkpoint, "checkpoint")
    )
    description = _actor_description(checkpoint)
    observed = _schema_method(description["schemas"])
    if observed != expected_method:
        raise ValueError(
            f"This entry evaluates {expected_method}; the checkpoint is {observed}"
        )
    opponent = plain.get("opponent", "tdm-alpha")
    kind = _reference_kind(opponent, base_dir=base)
    opponent_description = None
    if kind == "artifact":
        opponent = str(path(opponent, "opponent"))
        opponent_description = _actor_description(opponent)
    elif kind == "released":
        opponent = _saved_reference(opponent, base_dir=base)
        opponent_description = _actor_description(opponent)
    elif kind == "unknown":
        raise ValueError(f"Unknown opponent reference: {opponent}")
    for name in ("output_dir", "resume_from"):
        if name in settings and settings[name] is not None:
            settings[name] = str(path(settings[name], name))
    if "resume_from" in settings and settings["resume_from"] is None:
        raise ValueError("resume_from must name an exact saved result folder")
    if (
        settings.get("output_dir") is not None
        and settings.get("resume_from") is not None
    ):
        raise ValueError("Resume inherits its result folder; omit output_dir")
    if "num_episodes" not in settings and "resume_from" not in settings:
        raise ValueError("Fresh evaluation needs num_episodes")
    if "red_zone_depth" in settings:
        depth = settings["red_zone_depth"]
        if type(depth) is int:
            try:
                depth = float(depth)
            except OverflowError as error:
                raise ValueError("red_zone_depth must be finite") from error
        settings["red_zone_depth"] = _validate_red_zone_depth(depth)
    resources = _launch.resource_settings(plain.get("resources"), base_dir=base)
    checkpoint, opponent = _evaluation_paths(
        checkpoint,
        opponent,
        base_dir=base,
        launch_dir=launch_dir,
        output_dir=settings.get("output_dir"),
        resume_from=settings.get("resume_from"),
        compilation_cache=None
        if resources is None
        else cast(Mapping[str, Any], plain["resources"]).get("compilation_cache"),
    )
    launch = Path(_path(str(launch_dir), base, "launch_dir"))
    if resources is not None and (
        len(resources["devices"]) != 1 or resources["devices"][0]["max_workers"] != 1
    ):
        raise ValueError("An evaluation needs one device with max_workers=1")
    environment = _launch.resource_environment(
        None if resources is None else resources["devices"][0],
        compilation_cache=None
        if resources is None
        else resources.get("compilation_cache"),
    )
    request = {
        "expected_method": expected_method,
        "observed_method": observed,
        "checkpoint": str(checkpoint),
        "artifact": {
            key: value
            for key, value in _artifact_identity(description).items()
            if key != "metadata"
        },
        "opponent": opponent,
        "opponent_artifact": None
        if opponent_description is None
        else {
            key: value
            for key, value in _artifact_identity(opponent_description).items()
            if key != "metadata"
        },
        "evaluation": dict(settings),
        "resources": resources,
        "resource_environment": _launch.resource_facts(environment),
        "launch_dir": str(launch),
    }
    if plain.get("preview", False):
        report = {
            **request,
            "preview": True,
            "pending_checks": [
                "Payload verification",
                "Factory execution",
                "Device access",
                "Saved pass count when omitted",
                "Evaluation conditions and recovery",
            ],
        }
        print(json.dumps(report, indent=2, allow_nan=False))
        return report
    if resources is not None:
        if "jax" in sys.modules:
            raise RuntimeError("Set resources before importing numerical modules")
        _launch.check_resource_devices(resources)
        os.environ.update(environment)
    records = launch / ".hydra"
    if (records / "evaluation_request.json").exists():
        raise ValueError(f"Launch already has evaluation settings: {records}")
    result = None
    try:
        import marl_battlegrounds as marl_bgs

        if "num_episodes" not in settings:
            saved = marl_bgs.load_results(
                settings["resume_from"],
                phase=settings.get("phase", "evaluation"),
                pass_id=settings.get("pass_id", "1"),
            )
            passes = saved.metadata["passes"]
            if len(passes) != 1:
                raise ValueError("Resume needs one exact saved evaluation pass")
            count = next(iter(passes.values())).get("details", {}).get("num_episodes")
            if type(count) is not int or count <= 0:
                raise ValueError("The selected pass has no valid saved num_episodes")
            settings["num_episodes"] = count
        request["effective_evaluation"] = dict(settings)
        records.mkdir(parents=True, exist_ok=True)
        atomic_json(records / "evaluation_request.json", request)
        with _method_loading_scope():
            actor = marl_bgs.load_method(str(checkpoint))
            other = marl_bgs.load_method(opponent)
        for reference, loaded, original in (
            (checkpoint, actor, description),
            (opponent, other, opponent_description),
        ):
            if original is None:
                continue
            if (
                getattr(loaded, "checkpoint", None)
                != _artifact_identity(original)["actor_digest"]
                or _actor_description(reference)["checkpoint_id"]
                != original["checkpoint_id"]
            ):
                raise ValueError("Saved actor identity changed during evaluation setup")
        result = marl_bgs.evaluate(actor, other, **settings)
        atomic_json(
            records / "evaluation_result.json",
            {
                "run_dir": None if result.run_dir is None else str(result.run_dir),
                "status": result.status,
                "run_id": result.metadata.get("run_id"),
                "phase": result.metadata.get("phase"),
                "pass_id": result.metadata.get("pass_id"),
            },
        )
    except Exception as error:
        if result is not None and result.run_dir is not None:
            error.add_note(f"Evaluation output: {result.run_dir}")
            print(f"Evaluation output: {result.run_dir}", file=sys.stderr)
        print(f"Evaluation launch records: {launch}", file=sys.stderr)
        raise
    print(f"Evaluation output: {result.run_dir or 'No game results saved'}")
    print(f"Evaluation launch records: {launch}")
    return result


def evaluation_main(expected_method: str) -> int:
    """Compose packaged evaluation YAML and run the fixed method's shared adapter.

    expected_method is set by the thin script, never by an override. Use --help,
    --cfg job or preview=true to inspect inputs; --config-name evaluate_resume
    selects sparse resume settings. Return zero after success; retain Hydra/API
    failures and their result-directory notes. The original cwd owns all paths.
    """
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd

    @hydra.main(version_base="1.3", config_path="configs", config_name="evaluate")
    def configured(config: DictConfig) -> None:
        """Call the shared adapter once with Hydra's original directory."""
        previous = Path.cwd()
        try:
            os.chdir(get_original_cwd())
            run_evaluation(
                config,
                expected_method=expected_method,
                base_dir=get_original_cwd(),
                launch_dir=HydraConfig.get().runtime.output_dir,
            )
        finally:
            os.chdir(previous)

    prepare_hydra()
    configured()
    return 0
