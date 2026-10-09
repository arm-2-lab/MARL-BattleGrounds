"""Resolve method references: built-in names, saved actors, models and factories.

Command factories and tournament loaders share the ``module:function`` grammar
through installed_callable. load_method is the one owner of the wider
reference grammar used for training opponents: a built-in Policy name, an
actor export or complete learner checkpoint directory, a portable model file,
a released model (a typed name such as ``MARL-BattleGrounds/<repo>@100M``, an
exact ``hf://`` address or a local released-layout folder; released_models owns
those rules), or a ``module:function`` factory. Importing this module uses only
the standard library; load_method imports the evaluation, training, portable
model and download code it needs only when called. Calling a selected function
remains the caller's responsibility; imported researcher code may have its own
side effects.
"""

from __future__ import annotations

# These private setup helpers are shared by training and command owners.
# pyright: reportUnusedFunction=false
import importlib
import inspect
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from marl_battlegrounds.policies import (
    BUILTIN_POLICY_NAMES as BUILTIN_POLICY_NAMES,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.policy_execution import Policy, System


# Each training or evaluation setup owns this short-lived cache. Context-local storage
# keeps concurrent callers separate and does not retain anything between runs.
_setup_inputs: ContextVar[
    tuple[dict[str, Policy | System], dict[str, dict[str, Any]]] | None
] = ContextVar("marl_training_setup_inputs", default=None)


@contextmanager
def _method_loading_scope() -> Generator[None]:
    """Reuse reference loads during one method setup, then release them.

    Each exact resolved reference is loaded once during setup. Role-specific
    freezing and resource scopes still belong to collection/evaluation. Factories
    should return fixed methods and support the System nested-scope contract.
    A nested setup gets its own cache. Always reset on error or exit;
    callers release setup references before their training or evaluation loop.
    """
    token = _setup_inputs.set(({}, {}))
    try:
        yield
    finally:
        _setup_inputs.reset(token)


def _release_setup_inputs() -> None:
    """Release cached setup objects before training; retained roles own their data.

    Clear the current attempt's method and identity dictionaries, then disable
    caching for the remaining execution. This does not close resource scopes or
    alter live method objects. Outside trainer setup it is a no-op.
    """
    cached = _setup_inputs.get()
    if cached is not None:
        cached[0].clear()
        cached[1].clear()
        _setup_inputs.set(None)


def _reference_kind(
    reference: str, *, base_dir: Path | None = None
) -> Literal["builtin", "factory", "artifact", "released", "unknown"]:
    """Classify a method reference without importing code or reading payloads.

    Built-in names win. A bare module:function is a factory even beside a folder
    with that name. Next come released models (released_models._released_form):
    a local released-layout folder (optionally with an ``@`` step), and, when no
    local path with that exact text exists, an ``hf://`` address or a typed name
    such as ``MARL-BattleGrounds/<repo>@100M``. These are "released". Explicit
    paths (including ./model:version) are artifacts; other existing bare folders
    and bare portable model files use base_dir, or the current directory.
    Missing plain names stay unknown so the loading owner can report a useful
    error. Reject nontext and blank references. This checks path existence and
    reads a bare existing file's ZIP index only; no network call is made.
    """
    from marl_battlegrounds.released_models import (
        _released_form,  # pyright: ignore[reportPrivateUsage]
    )

    if not isinstance(cast(object, reference), str):
        raise TypeError("A method reference must be a string")
    if not reference.strip():
        raise ValueError("A method reference must not be empty")
    if reference in BUILTIN_POLICY_NAMES:
        return "builtin"
    if ":" in reference and "/" not in reference and not reference.startswith("."):
        return "factory"
    if _released_form(reference, base_dir) is not None:
        return "released"
    path = Path(reference).expanduser()
    if path.is_absolute() or "/" in reference or reference.startswith("."):
        return "artifact"
    local = (Path.cwd() if base_dir is None else base_dir) / path
    if local.is_dir():
        return "artifact"
    if local.is_file():
        from marl_battlegrounds.portable_models import is_portable_model_file

        if is_portable_model_file(local):
            return "artifact"
    return "unknown"


def _method_output_paths(
    references: Sequence[object],
    *,
    output_dir: str | Path | None,
    resume_from: str | Path | None,
    additional_outputs: Sequence[str | Path] = (),
    result_children: Sequence[str] = (),
) -> tuple[Path | None, Path | None]:
    """Keep output and resume paths separate from known saved method inputs.

    Classify string references through the same loader grammar, without loading
    arrays or calling factories. Live methods retain caller-owned input handling.
    Reject lexical symbolic links and writes under an input artifact. Extra
    launch/cache outputs must also stay outside the result or resumed run. Return
    absolute output/resume paths; create no files and initialize no backend.
    result_children names known child folders below a fresh output parent; check
    their lexical paths too, so they cannot alias or contain model inputs.
    """
    from marl_battlegrounds.evaluation.revision import (
        _check_output_path,  # pyright: ignore[reportPrivateUsage]
    )

    def path(value: str | Path) -> Path:
        """Reject symbolic aliases before resolving the caller's output path."""
        target = Path(value).expanduser().absolute()
        if any(part.is_symlink() for part in (*target.parents, target)):
            raise ValueError(
                "Method input and output paths must not use symbolic links"
            )
        return target.resolve()

    if output_dir is not None and resume_from is not None:
        raise ValueError("Resume inherits its result folder; omit output_dir")
    inputs = [
        path(value)
        for value in references
        if isinstance(value, str) and _reference_kind(value) == "artifact"
    ]
    output = None if output_dir is None else path(output_dir)
    resume = None if resume_from is None else path(resume_from)
    results = [value for value in (output, resume) if value is not None]
    if output is not None:
        results.extend(path(output / child) for child in result_children)
    extras = [path(value) for value in additional_outputs]
    for target in (*results, *extras):
        if any(target == source or source in target.parents for source in inputs):
            raise ValueError("Evaluation outputs must stay outside input artifacts")
        _check_output_path(target)
    for target in extras:
        if any(target == result or result in target.parents for result in results):
            raise ValueError("Launch and cache records must stay outside results")
    return output, resume


def installed_callable(reference: str) -> Callable[..., object]:
    """Import a trusted ``module:function`` and return it without calling it.

    The attribute must be one module-level name, not a dotted traversal.
    Invalid names and noncallable attributes raise ValueError. Import and
    attribute errors retain their original cause. No files are downloaded and
    no path is added to Python's import search path.
    """
    if not isinstance(cast(object, reference), str) or reference.count(":") != 1:
        raise ValueError("Controller factory/loader must use module:function")
    module_name, attribute = reference.split(":")
    if not module_name or not attribute or "." in attribute:
        raise ValueError(
            "Controller factory/loader must name one module-level callable"
        )
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise ValueError("Controller factory/loader is not callable")
    return value


def load_factory(reference: str) -> Policy | System:
    """Import a ``module:function`` factory, call it once, and check what it returns.

    Parameters
    ----------
    reference : str
        ``module:function`` naming one module-level callable that takes no
        arguments.

    Returns
    -------
    Policy or System
        The factory's result, unfrozen.

    Raises
    ------
    ValueError
        The text is not a valid ``module:function`` reference or names a
        noncallable attribute.
    TypeError
        The factory returned something other than a Policy or System.
    ImportError, AttributeError
        The module or attribute cannot be imported, with the original cause.
        Any error raised by the factory itself propagates with its own type.

    Notes
    -----
    Host-only. Runs researcher code once. The command line and load_method
    share this one rule.
    """
    from marl_battlegrounds.evaluation.policy_execution import Policy, System

    value = installed_callable(reference)()
    if not isinstance(value, (Policy, System)):
        raise TypeError(f"Factory {reference!r} must return a System or Policy")
    return value


def _reference_key(reference: str) -> str:
    """Return one setup-cache key, rejecting linked actor paths before collapse.

    Factories and built-ins keep their exact reference. Artifact directories use
    the existing checkpoint path guard before resolving dot/parent components.
    Equivalent ordinary paths then share loads and receipts; a symbolic-link
    alias can never bypass the guard through a prior cached object. A portable
    model file uses its fully resolved path (its content identity is its hash).
    A released model uses its saved form (released_models._saved_reference):
    the exact ``hf://`` address, or the resolved path of the chosen local file,
    so a typed name and its address share one load. Finding that form may
    download the repo's small index files.
    """
    kind = _reference_kind(reference)
    if kind == "released":
        from marl_battlegrounds.released_models import (
            _saved_reference,  # pyright: ignore[reportPrivateUsage]
        )

        saved = _saved_reference(reference)
        return saved if saved.startswith("hf://") else str(Path(saved).resolve())
    if kind != "artifact":
        return reference
    path = Path(reference).expanduser()
    if path.is_file():
        return str(path.resolve())
    from marl_battlegrounds.training._checkpoint_metadata import (
        _directory,  # pyright: ignore[reportPrivateUsage]
    )

    return str(_directory(path).resolve())


def load_method(reference: str) -> Policy | System:
    """Turn one method reference string into a Policy or System, unfrozen.

    Parameters
    ----------
    reference : str
        Checked in this order. A built-in Policy name ("random", "tdm-alpha",
        "tdm-beta" or "tdm-gamma", as ``policy`` defines them) gives that
        Policy. A bare module:function invokes that factory, even beside a folder
        with the same name. A released model comes next: a local folder in the
        released layout (``steps.csv`` and ``config.json``, for example after
        ``hf download --local-dir``), optionally with an ``@`` step; then, when
        no local path with that exact text exists, an exact address
        ``hf://<owner>/<repo>@<commit>/<file>`` or a typed name such as
        ``"MARL-BattleGrounds/tdm-season-0-qmix@100M"`` (no suffix: the entered
        save; ``@final``: the last save; ``@<N>M``: the nearest save at or
        before N million steps; ``@99_950_592``: exactly that save). A
        portable model file (a path to one ``.zip`` model file) loads through
        ``marl_battlegrounds.portable_models``; neither form needs the training
        extra. Explicit paths and other existing folders load through
        ``marl_battlegrounds.training.load_system``. Actor exports and complete
        learner checkpoints are supported, including historical actor inputs.
        Only actor arrays are restored; every payload file is checked. This needs
        the training extra and grants no permission to resume training.
        Spell colon-containing paths with ./ or an absolute path.
        A relative path resolves against the current directory,
        so saved configurations must use absolute paths.

    Returns
    -------
    Policy or System
        The method exactly as its source produced it. Pass it to evaluate,
        team, a declared population or the training API; those owners freeze
        inputs when their workflow requires it. This call alone plays no game.
        BattleClient retains borrowed objects and their parameters for its
        blocking lifetime; the caller must keep those parameters fixed.

    Raises
    ------
    TypeError
        reference is not a string, or a factory returned something other than
        a Policy or System.
    ValueError
        The text is empty, names no built-in, directory, model file, released
        model or valid factory, the directory is incomplete, or its files fail
        their checks. For a released model: the step does not exist (the
        message lists the nearest saves), the ``@`` suffix is not a step, the
        repo does not exist, a downloaded file's SHA-256 differs from its
        repo's ``steps.csv``, the address names an original actor folder in a
        dataset (download it to a local folder first), or Hugging Face is
        offline and the file is not cached (the message gives the
        ``models download`` command).
    OSError
        An actor directory or model file cannot be read.
    ImportError
        A directory was given without the training extra installed, or a
        factory's module cannot be imported. Errors from a factory's call
        propagate with their original type.

    Notes
    -----
    Host-only. A directory is read and its file hashes are checked; a factory
    runs researcher code once. The trainer reuses each resolved reference during
    its setup; ordinary independent calls do not share a cache. Local paths and
    factories write nothing. A released model is downloaded once into Hugging
    Face's standard cache (``HF_HOME``, ``HF_HUB_CACHE``) when not already
    there; a cached pinned model needs no network call. An unpinned repo uses
    its latest commit with a warning. To record the exact file, save the form
    ``released_models`` gives (the trainer and tournaments do this).
    """
    kind = _reference_kind(reference)
    cached = _setup_inputs.get()
    model_file: Path | None = None
    if kind == "released":
        from marl_battlegrounds.released_models import (
            _fetch,  # pyright: ignore[reportPrivateUsage]
        )

        save = _fetch(reference)
        model_file = save.path
        key = (
            save.reference
            if save.reference.startswith("hf://")
            else str(Path(save.reference).resolve())
        )
    else:
        key = _reference_key(reference)
    if cached is not None and key in cached[0]:
        return cached[0][key]
    if kind == "builtin":
        from marl_battlegrounds.evaluation.policy_execution import policy

        value = policy(reference)
    elif kind == "factory":
        value = load_factory(reference)
    elif kind in ("artifact", "released"):
        path = Path(reference).expanduser() if model_file is None else model_file
        portable = model_file is not None
        if not portable and path.is_file():
            from marl_battlegrounds.portable_models import is_portable_model_file

            portable = is_portable_model_file(path)
        if portable:
            from marl_battlegrounds.portable_models import load_portable_model

            value = load_portable_model(path)
        else:
            from marl_battlegrounds.training.checkpoints import load_system

            value = load_system(path)
    else:
        raise ValueError(
            f"Method reference {reference!r} is not a built-in name, an existing "
            "actor export/checkpoint directory, a model file, a released model "
            "name or address, or a module:function factory"
        )
    if cached is not None:
        cached[0][key] = value
    return value


def _reference_identity(reference: str) -> dict[str, Any]:
    """Read a reload identity once per trainer setup, returning a separate mapping.

    The uncached reader owns artifact, built-in and trusted factory evidence.
    Outside the trainer's scope every call reads current input facts. Cached
    records contain only JSON data; copies prevent a caller from editing the
    identity another role receives. No payload or factory body is loaded here.
    """
    import copy

    cached = _setup_inputs.get()
    if cached is None:
        return _read_reference_identity(reference)
    key = _reference_key(reference)
    if key not in cached[1]:
        cached[1][key] = _read_reference_identity(reference)
    return copy.deepcopy(cached[1][key])


def _read_reference_identity(reference: str) -> dict[str, Any]:
    """Describe one reloadable input without calling a factory or loading arrays.

    Saved artifacts use their checked description identity. Built-ins use their
    installed controller descriptor; random uses its shared callable evidence.
    Factories record function evidence, defining-file bytes when available and
    empty explicit kwargs. Declared __wrapped__ chains include each wrapper and
    wrapped function; malformed chains fail before a factory can run. This imports
    trusted factory code but never calls it.
    Mutable globals, closure values and remote provider state remain unknown.
    The returned identity excludes filesystem locations so a byte-identical
    actor or defining file can move. Portable model files and released models
    record ``{"kind": "portable_model", "sha256": <file SHA-256>}``, so a run can
    move between a released name, its ``hf://`` address and the same file on
    disk. A released model's hash comes from its repo's ``steps.csv`` (small
    index downloads only); a local file is hashed. Unknown reference names are
    rejected.
    """
    kind = _reference_kind(reference)
    if kind == "released":
        from marl_battlegrounds.released_models import (
            _identity,  # pyright: ignore[reportPrivateUsage]
        )

        return _identity(reference)
    if kind == "artifact":
        path = Path(reference).expanduser()
        if path.is_file():
            from marl_battlegrounds.portable_models import is_portable_model_file
            from marl_battlegrounds.released_models import (
                _sha256,  # pyright: ignore[reportPrivateUsage]
            )

            if not is_portable_model_file(path):
                raise ValueError(f"{reference!r} is not a portable model file")
            return {"kind": "portable_model", "sha256": _sha256(path)}
        from marl_battlegrounds.training._checkpoint_metadata import (
            read_checkpoint_description,
        )

        details = read_checkpoint_description(path)
        return {"kind": "artifact", "checkpoint_id": details["checkpoint_id"]}
    from marl_battlegrounds.evaluation.recording_identity import (
        _callable_evidence,  # pyright: ignore[reportPrivateUsage]
    )

    if kind == "builtin":
        from marl_battlegrounds.evaluation.policy_execution import (
            controller_identity,
            policy,
        )

        chosen = policy(reference)
        return {
            "kind": "builtin",
            "name": reference,
            "controller": controller_identity(chosen),
            "callable": _callable_evidence(chosen.apply),
        }
    if kind != "factory":
        raise ValueError(f"Unknown method reference: {reference!r}")
    function = installed_callable(reference)
    callables: list[dict[str, Any]] = []
    seen: set[int] = set()
    while True:
        if not callable(function):
            raise TypeError(
                f"Factory {reference!r} has a non-callable __wrapped__ value"
            )
        if id(function) in seen or len(callables) >= 64:
            raise ValueError(
                f"Factory {reference!r} has a cyclic or excessively long "
                "__wrapped__ chain"
            )
        seen.add(id(function))
        try:
            source_file = inspect.getsourcefile(function)
        except TypeError:
            source_file = None
        callables.append(
            {
                "callable": _callable_evidence(function),
                "defining_file_sha256": (
                    sha256(Path(source_file).read_bytes()).hexdigest()
                    if source_file is not None and Path(source_file).is_file()
                    else None
                ),
            }
        )
        if not hasattr(function, "__wrapped__"):
            break
        function = getattr(function, "__wrapped__")  # noqa: B009
    identity: dict[str, Any] = {
        "kind": "factory",
        "reference": reference,
        **callables[0],
        "kwargs": {},
        "external_state": "unknown",
    }
    if len(callables) > 1:
        identity["wrapped_callables"] = callables[1:]
    return identity


def validate_saved_method_reference(reference: str) -> None:
    """Check a durable method reference without opening files or calling factories.

    reference is a nonempty built-in Policy name, an absolute actor
    export/checkpoint or model file path, an exact released address
    (``hf://<owner>/<repo>@<40-character commit>/<file>`` for a model file or
    ``hf://datasets/<owner>/<repo>@<commit>/<folder>`` for a dataset folder), or
    module:function. Relative paths, including bare directory names, are refused
    because their meaning changes with the working directory; so are typed
    released names, whose files can change (the trainer saves their address).
    Built-in names use the existing Policy authority. Missing imports, artifact
    files and downloads are checked later by load_method. Invalid types or
    grammar raise TypeError or ValueError.
    """
    if not isinstance(cast(object, reference), str):
        raise TypeError("A saved method reference must be a string")
    if not reference.strip():
        raise ValueError("A saved method reference must not be empty")
    if reference in BUILTIN_POLICY_NAMES:
        return
    if reference.startswith("hf://"):
        from marl_battlegrounds.released_models import (
            _parse_address,  # pyright: ignore[reportPrivateUsage]
        )

        _parse_address(reference)
        return
    if Path(reference).is_absolute():
        return
    if reference.count(":") == 1:
        module, attribute = reference.split(":")
        if module and attribute and "." not in attribute and "/" not in reference:
            return
    raise ValueError(
        "Method paths must be absolute; otherwise use a built-in name, "
        "an hf:// address or module:function"
    )
