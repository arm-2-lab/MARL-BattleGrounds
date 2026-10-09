"""Find, download and check released MARL-BattleGrounds models by name.

This module is the one owner of released names. A researcher types one name
anywhere MARL-BGs takes a model, for example
``MARL-BattleGrounds/tdm-season-0-qmix@100M``, and gets that exact saved model
from Hugging Face (HF), downloaded once into HF's standard cache and checked.

What people type (a "typed name"):

- ``MARL-BattleGrounds/<repo>``: the save the model entered its season with
  (``entered_step`` in the repo's ``config.json``).
- ``MARL-BattleGrounds/<repo>@final``: the last save.
- ``MARL-BattleGrounds/<repo>@100M``: the nearest save at or before
  100,000,000 steps.
- ``MARL-BattleGrounds/<repo>@100_139_008`` or ``@100139008``: exactly that save.

Any other ``@`` suffix, such as a commit hash or a branch name, is refused.
Typed names belong to the MARL-BattleGrounds organization (any letter case),
so an ordinary relative path such as ``actors/policy`` keeps its path meaning
even before it exists. A repo elsewhere on HF that uses the same layout loads
through its exact ``hf://`` address.
A local folder in the released layout (it holds ``steps.csv`` and
``config.json``, for example after ``hf download --local-dir``) takes the same
``@`` rules. A local path that exists always wins over a typed name.

What gets saved: wherever MARL-BGs saves a model reference, a typed name becomes
HF's exact file address, ``hf://<owner>/<repo>@<40-character commit>/<file>``,
and a released-layout folder becomes the absolute path of the chosen file.
Original actor folders in a released dataset use
``hf://datasets/<owner>/<repo>@<commit>/<folder>``. ``load_method`` accepts every
model form; a dataset folder is used from a local copy (see
_dataset_folder_error). Resume and reproduction always use the recorded commit.

Public entry point: download_models, the prefetch used by
``python -m marl_battlegrounds models download <name>...``. The other helpers
are private and shared by the package's loading and saving owners
(_method_loading, training, evaluation, experiments and the BattleClient).
Importing this module uses only the standard library; huggingface_hub is
imported only when a name must be resolved or a file downloaded.
"""

from __future__ import annotations

# The private helpers are shared by the loading and saving owners.
# pyright: reportUnusedFunction=false
import csv
import hashlib
import io
import json
import os
import re
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from types import ModuleType

RELEASE_ORGANIZATION = "MARL-BattleGrounds"
"""The Hugging Face organization that hosts the released models and data."""

RESULTS_DATASET = "MARL-BattleGrounds/tdm-season-0-results"
"""The dataset repo with Season 0 results and the original opponent folders."""

TRAINING_OPPONENTS = "training-opponents"
"""Folder in RESULTS_DATASET with the training and validation opponent bundle."""

PINNED_COMMITS: dict[str, str] = {
    "MARL-BattleGrounds/tdm-season-0-recurrent-ippo": (
        "48a062ae79167835fb81dfc906bd3aa1711f4483"
    ),
    "MARL-BattleGrounds/tdm-season-0-recurrent-mappo": (
        "b27435d2b153817b1baa204f5751303c9bf7ff60"
    ),
    "MARL-BattleGrounds/tdm-season-0-feedforward-ippo": (
        "ffdbd94424959854cad687186037464789e12ef7"
    ),
    "MARL-BattleGrounds/tdm-season-0-feedforward-mappo": (
        "aea78b3d6122932b92ca0c24a57932e283962f02"
    ),
    "MARL-BattleGrounds/tdm-season-0-pqn-vdn": (
        "34d7075272c7e95cba12c6a8dd8ff2b72b30ce80"
    ),
    "MARL-BattleGrounds/tdm-season-0-qmix": "73f2a413d2b214d7d8ac41c404da87965bea6f7a",
    RESULTS_DATASET: "4b8844694a2c5fe4d8fa5cdbb3b70a14afdb1680",
}
"""Repo id to the full 40-character commit of its release.

The six Season 0 model repos and the results dataset (RESULTS_DATASET).
Keys use HF's exact spelling; lookups ignore case. A repo listed here resolves
to this commit with no network call when its files are already cached. A repo
missing here resolves its latest ``main`` commit once per process, with a
warning that it is not pinned; saved records keep that commit either way.
"""

SUPPORTED_FORMATS = frozenset({"portable-model-1"})
"""The ``format`` values in ``steps.csv`` that this package can load."""

_LIBRARY = "marl-battlegrounds"
_PART = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
_NAME = re.compile(rf"(?P<repo>{_PART}/{_PART})(?:@(?P<suffix>.*))?", re.DOTALL)
_ADDRESS = re.compile(
    rf"hf://(?P<dataset>datasets/)?(?P<repo>{_PART}/{_PART})"
    r"@(?P<commit>[^/]*)/(?P<path>.+)",
    re.DOTALL,
)
_COMMIT = re.compile(r"[0-9a-f]{40}")
_EXACT = re.compile(r"\d+|\d{1,3}(?:_\d{3})+")
_MILLIONS = re.compile(r"(\d+)[Mm]")
_RECORD_FILES = ("run_details.json", "status.json", "validation_results.json")
# Latest main commit per lowercased repo id, resolved once per process.
_MAIN_COMMITS: dict[str, tuple[str, str]] = {}
_PACKAGE_FOLDER = os.path.dirname(__file__)


@dataclass(frozen=True)
class _Save:
    """One released save chosen from a name, an address or a released folder.

    Attributes
    ----------
    reference : str
        The saved form: an ``hf://`` address, or the absolute path of a file
        chosen from a local released-layout folder.
    step : int or None
        The save's training step (environment transitions). None for a
        dataset folder address.
    sha256 : str or None
        The model file's SHA-256 from ``steps.csv``. None for a dataset folder.
    path : Path or None
        The checked local model file after a fetch; None when only resolved.
    """

    reference: str
    step: int | None
    sha256: str | None
    path: Path | None = None


@dataclass(frozen=True)
class _Address:
    """One parsed ``hf://`` address: repo id, commit, path and repo kind."""

    repo: str
    commit: str
    path: str
    dataset: bool


def _hub() -> ModuleType:
    """Import huggingface_hub only when a download or a lookup is needed."""
    from importlib import import_module

    try:
        # The two submodules become attributes of the package for later use.
        hub = import_module("huggingface_hub")
        import_module("huggingface_hub.constants")
        import_module("huggingface_hub.errors")
    except ImportError as error:  # pragma: no cover - a base dependency
        raise ImportError(
            "Released models need the huggingface_hub package; "
            "reinstall marl-battlegrounds"
        ) from error
    return hub


def _repo_part_ok(repo: str) -> bool:
    """Apply HF's extra repo-id rule: no "--" and no ".." inside either part."""
    return "--" not in repo and ".." not in repo and not repo.endswith(".git")


def _split_name(text: str) -> tuple[str, str | None] | None:
    """Split typed-name text into (repo id, step suffix), or None if not a name.

    Parameters
    ----------
    text : str
        Text such as ``"MARL-BattleGrounds/tdm-season-0-qmix@100M"``.

    Returns
    -------
    tuple of (str, str or None), or None
        The repo id and the text after ``@`` (None when there is no ``@``).
        None means the text is not ``MARL-BattleGrounds/<repo>[@...]`` (the
        organization in any letter case), so it cannot be a typed name. The
        suffix is not checked here.
    """
    match = _NAME.fullmatch(text)
    if (
        match is None
        or not _repo_part_ok(match["repo"])
        or match["repo"].split("/")[0].lower() != RELEASE_ORGANIZATION.lower()
    ):
        return None
    return match["repo"], match["suffix"]


def _check_suffix(suffix: str | None, text: str) -> None:
    """Refuse a typed-name ``@`` suffix that is not one of the step forms.

    Accepted: no suffix, ``final``, ``<N>M`` (``final`` and ``M`` in any
    letter case) and an exact step written as plain digits or in groups of
    three with underscores (``100_139_008``).
    Raises ValueError naming text and the accepted forms otherwise, for
    example for a commit hash or a branch name.
    """
    if suffix is None or suffix.lower() == "final":
        return
    if _MILLIONS.fullmatch(suffix) or _EXACT.fullmatch(suffix):
        return
    raise ValueError(
        f"In the released name {text!r}, @ must choose a training step: "
        "@final, @<N>M (for example @100M) or an exact step such as "
        "@100_139_008. Commits and branch names are not accepted after @; "
        "to use an exact commit, give the full address "
        "hf://<owner>/<repo>@<commit>/<file>"
    )


def _typed_name(text: str) -> tuple[str, str | None] | None:
    """Check a typed name's spelling only; never touch files or the network.

    Parameters
    ----------
    text : str
        Any method reference text.

    Returns
    -------
    tuple of (str, str or None), or None
        (repo id, step suffix) when text is spelled as a typed released name,
        such as ``"MARL-BattleGrounds/tdm-season-0-qmix@final"``. None when it
        is not shaped like one (for example a built-in name, an absolute path,
        ``./folder``, ``actors/policy`` or an ``hf://`` address).

    Raises
    ------
    ValueError
        text has the typed-name shape but an ``@`` suffix that is not a step
        (see _check_suffix).

    Notes
    -----
    TrainConfig uses this for its model fields. A relative path that starts
    with a ``MARL-BattleGrounds/`` folder also has this shape; the trainer
    later finds whether it exists locally, and a local relative path is still
    refused there.
    """
    split = _split_name(text)
    if split is not None:
        _check_suffix(split[1], text)
    return split


def _parse_address(text: str) -> _Address:
    """Parse and check one saved ``hf://`` address.

    Parameters
    ----------
    text : str
        ``hf://<owner>/<repo>@<commit>/<file>`` for a model file, or
        ``hf://datasets/<owner>/<repo>@<commit>/<folder>`` for a dataset folder.
        The commit must be HF's full 40-character lowercase hexadecimal id.

    Returns
    -------
    _Address
        The repo id, commit, path inside the repo and whether it is a dataset.

    Raises
    ------
    ValueError
        The text is not that shape, the commit is not a full commit id, or the
        path is empty, uses ``.``, ``..`` or empty parts, or a backslash.
    """
    match = _ADDRESS.fullmatch(text)
    if match is None or not _repo_part_ok(match["repo"]):
        raise ValueError(
            f"{text!r} is not a released-model address; use "
            "hf://<owner>/<repo>@<commit>/<file> or "
            "hf://datasets/<owner>/<repo>@<commit>/<folder>"
        )
    if not _COMMIT.fullmatch(match["commit"]):
        raise ValueError(
            f"{text!r} must name a full 40-character commit after @, so the "
            "saved reference always means the same file"
        )
    path = match["path"].rstrip("/")
    parts = path.split("/")
    if "\\" in path or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{text!r} has an invalid path inside its repo")
    return _Address(match["repo"], match["commit"], path, bool(match["dataset"]))


def _is_released_folder(path: Path) -> bool:
    """Return whether path is a local folder in the released layout."""
    return (
        path.is_dir()
        and (path / "steps.csv").is_file()
        and (path / "config.json").is_file()
    )


def _released_form(
    reference: str, base_dir: Path | None = None
) -> Literal["address", "folder", "name"] | None:
    """Say which released form a reference has, using only cheap file checks.

    Parameters
    ----------
    reference : str
        A method reference string.
    base_dir : Path or None, default=None
        Folder that relative paths use; None means the current directory.

    Returns
    -------
    {"address", "folder", "name"} or None
        "folder" for a local released-layout folder, with or without an ``@``
        suffix. Otherwise, when no local path with this exact text exists:
        "address" for text starting with ``hf://`` and "name" for typed-name
        text. None for everything else, including every local path that
        exists and is not a released-layout folder.

    Notes
    -----
    A local path that exists always wins, as in HF's ``from_pretrained``.
    Reads no file contents and makes no network call. A dataset folder address
    is an "address" here, but loading it needs a local copy
    (see _dataset_folder_error).
    """
    base = Path.cwd() if base_dir is None else base_dir
    path = Path(reference).expanduser()
    local = path if path.is_absolute() else base / path
    if local.exists():
        return "folder" if _is_released_folder(local) else None
    if reference.startswith("hf://"):
        return "address"
    stem_text = reference.rpartition("@")[0]
    if stem_text:
        stem = Path(stem_text).expanduser()
        if _is_released_folder(stem if stem.is_absolute() else base / stem):
            return "folder"
    return "name" if _split_name(reference) is not None else None


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of one file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prefetch_hint(reference: str) -> str:
    """Return the command that downloads reference before an offline run."""
    return f"python -m marl_battlegrounds models download {reference}"


def _commit_for(
    repo: str,
    commits: Mapping[str, str | None] | None,
    *,
    dataset: bool = False,
    reference: str,
) -> tuple[str, str]:
    """Choose the commit a typed name uses: recorded, pinned or latest main.

    Parameters
    ----------
    repo : str
        Repo id as typed; case is ignored.
    commits : mapping or None
        Commits a saved run recorded (from _recorded_commits). They win over
        the package pins, so a resumed run keeps its files. A repo mapped to
        None was recorded at several commits and is refused.
    dataset : bool, default=False
        True for a dataset repo (only used for the latest-commit lookup).
    reference : str
        The original text, for error messages.

    Returns
    -------
    tuple of (str, str)
        HF's spelling of the repo id and the full commit.

    Raises
    ------
    ValueError
        The saved run recorded several commits of this repo, the repo does not
        exist on HF (or needs a login token), or HF is offline and the repo is
        not pinned.

    Warns
    -----
    UserWarning
        The repo is not pinned by this package, so its latest commit is used.

    Notes
    -----
    The latest-commit lookup is the only network call here, done once per
    process and repo. Other network errors from huggingface_hub propagate
    with their own types.
    """
    key = repo.lower()
    for known in (commits or {}), PINNED_COMMITS:
        for name, commit in known.items():
            if name.lower() == key:
                if commit is None:
                    raise ValueError(
                        f"The saved run records more than one commit of {name}; "
                        "give the exact hf:// address instead of "
                        f"{reference!r}"
                    )
                return name, commit
    if key not in _MAIN_COMMITS:
        hub = _hub()
        if hub.constants.is_offline_mode():
            raise ValueError(
                f"{repo} is not pinned by this package, so finding its latest "
                "commit needs the network, but HF_HUB_OFFLINE is set. Run "
                f"`{_prefetch_hint(reference)}` while online and use the hf:// "
                "address it prints"
            )
        api = hub.HfApi(library_name=_LIBRARY)
        try:
            info = (
                api.dataset_info(repo, revision="main")
                if dataset
                else api.model_info(repo, revision="main")
            )
        except hub.errors.RepositoryNotFoundError as error:
            raise ValueError(
                f"{reference!r} is not a local path, and Hugging Face has no "
                f"released model repo {repo!r} (or it needs a login token)"
            ) from error
        _MAIN_COMMITS[key] = (str(info.id), str(info.sha))
        warnings.warn(
            f"{info.id} is not pinned by this package; using its latest commit "
            f"{info.sha}. Saved records keep this exact commit.",
            # Point at the caller's own line, outside this package.
            skip_file_prefixes=(_PACKAGE_FOLDER,),
        )
    return _MAIN_COMMITS[key]


def _hub_file(repo: str, filename: str, commit: str, *, reference: str) -> Path:
    """Return one model-repo file at an exact commit, from the HF cache if present.

    Parameters
    ----------
    repo, filename, commit : str
        Model repo id, path inside the repo and full commit.
    reference : str
        The original text, used in error messages and the prefetch hint.

    Returns
    -------
    Path
        The file in HF's standard cache (``HF_HOME`` and ``HF_HUB_CACHE``
        choose where). On Linux and macOS this is a link into HF's blob store.

    Raises
    ------
    ValueError
        The file is missing at that commit, the commit or repo does not exist,
        or HF is offline and the file is not cached (the message gives the
        prefetch command).

    Notes
    -----
    A cached file is used with no network call, even when HF is online.
    Otherwise ``huggingface_hub.hf_hub_download`` fetches it, with its own
    retries, resume, locking and ``HF_TOKEN``/``HF_ENDPOINT`` support.
    """
    hub = _hub()
    cached = hub.try_to_load_from_cache(repo, filename, revision=commit)
    if isinstance(cached, str):
        return Path(cached)
    errors = hub.errors
    try:
        return Path(
            hub.hf_hub_download(
                repo,
                filename,
                revision=commit,
                library_name=_LIBRARY,
            )
        )
    except (errors.LocalEntryNotFoundError, errors.OfflineModeIsEnabled) as error:
        raise ValueError(
            f"{reference!r} needs {repo}/{filename} at commit {commit}, which is "
            "not in the local Hugging Face cache, and Hugging Face cannot be "
            "reached (HF_HUB_OFFLINE or no network). Run "
            f"`{_prefetch_hint(reference)}` on a machine with network access first"
        ) from error
    except errors.EntryNotFoundError as error:
        raise ValueError(
            f"{reference!r}: {repo} has no file {filename} at commit {commit}"
        ) from error
    except errors.RevisionNotFoundError as error:
        raise ValueError(f"{reference!r}: {repo} has no commit {commit}") from error
    except errors.RepositoryNotFoundError as error:
        raise ValueError(
            f"{reference!r}: Hugging Face has no repo {repo} "
            "(or it needs a login token)"
        ) from error


@dataclass(frozen=True)
class _Index:
    """One repo's save list: entered step and every loadable model row."""

    entered_step: int | None
    rows: tuple[dict[str, Any], ...]


def _read_index(steps: bytes, config: bytes, where: str) -> _Index:
    """Parse ``steps.csv`` and ``config.json`` into the loadable model rows.

    Parameters
    ----------
    steps, config : bytes
        The two files' bytes.
    where : str
        Repo or folder name for error messages.

    Returns
    -------
    _Index
        ``entered_step`` from config.json (None when absent) and the rows whose
        ``kind`` is ``model`` and whose ``format`` is supported, sorted by step,
        each with integer ``step`` and ``bytes``.

    Raises
    ------
    ValueError
        A file is malformed, a needed column is missing, a step or SHA-256 is
        invalid, a file path is unsafe, or one step has two loadable rows.
    """
    try:
        details: object = json.loads(config)
        reader = csv.DictReader(io.StringIO(steps.decode("utf-8")))
        raw = list(reader)
    except (UnicodeDecodeError, ValueError, csv.Error) as error:
        raise ValueError(
            f"{where} has an unreadable steps.csv or config.json"
        ) from error
    if not isinstance(details, dict):
        raise ValueError(f"{where} config.json must be a JSON object")
    details = cast(dict[str, object], details)
    needed = {"step", "kind", "file", "bytes", "sha256", "format"}
    if not needed <= set(reader.fieldnames or ()):
        raise ValueError(f"{where} steps.csv lacks columns {sorted(needed)}")
    rows: dict[int, dict[str, Any]] = {}
    for row in raw:
        if row["kind"] != "model" or row["format"] not in SUPPORTED_FORMATS:
            continue
        file = row["file"]
        if (
            not row["step"].isdigit()
            or not row["bytes"].isdigit()
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or "\\" in file
            or any(part in ("", ".", "..") for part in file.split("/"))
        ):
            raise ValueError(f"{where} steps.csv has an invalid row: {row}")
        step = int(row["step"])
        if step in rows:
            raise ValueError(f"{where} steps.csv lists step {step:_} twice")
        rows[step] = {**row, "step": step, "bytes": int(row["bytes"])}
    entered = details.get("entered_step")
    if entered is not None and (type(entered) is not int or entered < 0):
        raise ValueError(f"{where} config.json has an invalid entered_step")
    return _Index(entered, tuple(rows[step] for step in sorted(rows)))


def _nearest(index: _Index, target: int) -> str:
    """List up to two saves on each side of target, written like 100_139_008."""
    steps = [row["step"] for row in index.rows]
    below = [step for step in steps if step < target][-2:]
    above = [step for step in steps if step > target][:2]
    return ", ".join(f"{step:_}" for step in (*below, *above)) or "none"


def _choose(index: _Index, suffix: str | None, text: str) -> dict[str, Any]:
    """Pick one save row by the step rules in this module's description.

    Parameters
    ----------
    index : _Index
        The repo's or folder's loadable saves.
    suffix : str or None
        The text after ``@``; None means the entered save.
    text : str
        The original reference, for error messages.

    Returns
    -------
    dict
        The chosen ``steps.csv`` row.

    Raises
    ------
    ValueError
        No loadable saves; no entered step; no save at or before an ``@<N>M``
        target; no save at an exact step (the message lists the nearest
        saves); or a suffix that is not a step.
    """
    _check_suffix(suffix, text)
    if not index.rows:
        raise ValueError(f"{text!r} has no saves this package can load")
    by_step = {row["step"]: row for row in index.rows}
    if suffix is None:
        if index.entered_step is None or index.entered_step not in by_step:
            raise ValueError(f"{text!r} names no entered save; add @final or a step")
        return by_step[index.entered_step]
    if suffix.lower() == "final":
        return index.rows[-1]
    millions = _MILLIONS.fullmatch(suffix)
    if millions is not None:
        target = int(millions[1]) * 1_000_000
        earlier = [row for row in index.rows if row["step"] <= target]
        if not earlier:
            raise ValueError(
                f"{text!r}: no save at or before step {target:_}; nearest saves: "
                f"{_nearest(index, target)}"
            )
        return earlier[-1]
    target = int(suffix.replace("_", ""))
    if target not in by_step:
        raise ValueError(
            f"{text!r}: no save at exactly step {target:_}; nearest saves: "
            f"{_nearest(index, target)}"
        )
    return by_step[target]


def _hub_index(repo: str, commit: str, reference: str) -> _Index:
    """Read a released repo's ``steps.csv`` and ``config.json`` at one commit."""
    steps = _hub_file(repo, "steps.csv", commit, reference=reference)
    config = _hub_file(repo, "config.json", commit, reference=reference)
    return _read_index(steps.read_bytes(), config.read_bytes(), f"{repo}@{commit}")


def _resolve(
    reference: str,
    *,
    base_dir: Path | None = None,
    commits: Mapping[str, str | None] | None = None,
) -> _Save:
    """Turn a released reference into its saved form, without the model file.

    Parameters
    ----------
    reference : str
        A typed name, an ``hf://`` address or a released-layout folder (with
        an optional ``@`` suffix).
    base_dir : Path or None, default=None
        Folder for relative local paths; None means the current directory.
    commits : mapping or None, default=None
        Commits recorded by a saved run, which typed names use first.

    Returns
    -------
    _Save
        Saved form, step and SHA-256; path is None.

    Raises
    ------
    ValueError
        reference is not a released form, or any rule of _commit_for,
        _hub_file, _read_index or _choose fails. A model address must name a
        file listed in that commit's ``steps.csv``.

    Notes
    -----
    May download the small ``steps.csv`` and ``config.json`` files and, for
    an unpinned repo, look up its latest commit once per process.
    """
    form = _released_form(reference, base_dir)
    if form == "address":
        address = _parse_address(reference)
        if address.dataset:
            return _Save(reference, None, None)
        repo = next(
            (name for name in PINNED_COMMITS if name.lower() == address.repo.lower()),
            address.repo,
        )
        index = _hub_index(repo, address.commit, reference)
        row = next((row for row in index.rows if row["file"] == address.path), None)
        if row is None:
            raise ValueError(
                f"{reference!r} is not a loadable model file in that commit's steps.csv"
            )
        canonical = f"hf://{repo}@{address.commit}/{address.path}"
        return _Save(canonical, row["step"], row["sha256"])
    if form == "folder":
        base = Path.cwd() if base_dir is None else base_dir
        local = Path(reference).expanduser()
        local = local if local.is_absolute() else base / local
        suffix = None
        if not _is_released_folder(local):
            stem, _, suffix = reference.rpartition("@")
            local = Path(stem).expanduser()
            local = local if local.is_absolute() else base / local
        index = _read_index(
            (local / "steps.csv").read_bytes(),
            (local / "config.json").read_bytes(),
            str(local),
        )
        row = _choose(index, suffix, reference)
        return _Save(str((local / row["file"]).resolve()), row["step"], row["sha256"])
    if form == "name":
        split = _typed_name(reference)
        assert split is not None
        repo, commit = _commit_for(split[0], commits, reference=reference)
        row = _choose(_hub_index(repo, commit, reference), split[1], reference)
        return _Save(f"hf://{repo}@{commit}/{row['file']}", row["step"], row["sha256"])
    raise ValueError(
        f"{reference!r} is not a released model name, hf:// address or "
        "released-layout folder"
    )


def _saved_reference(
    reference: str,
    *,
    base_dir: Path | None = None,
    commits: Mapping[str, str | None] | None = None,
) -> str:
    """Return the form to save for one method reference; the shared helper.

    Parameters
    ----------
    reference : str
        Any method reference string.
    base_dir : Path or None, default=None
        Folder for relative local paths; None means the current directory.
    commits : mapping or None, default=None
        Commits a saved run recorded (see _recorded_commits); typed names use
        them before the package pins, so resume never moves to newer files.

    Returns
    -------
    str
        A typed name becomes its exact ``hf://`` file address; a
        released-layout folder becomes the absolute path of the chosen file;
        an ``hf://`` address is checked and returned in HF's repo spelling.
        Every other reference (built-in names, factories, ordinary paths) is
        returned unchanged, so each caller keeps its own path handling.

    Raises
    ------
    ValueError
        See _resolve.

    Notes
    -----
    Never downloads a model file. An ``hf://`` address needs no download at
    all; a typed name reads its repo's small index files (from the cache when
    present). Prints nothing; the chosen step is in the returned file name. An
    unpinned repo gives a UserWarning.
    """
    form = _released_form(reference, base_dir)
    if form is None:
        return reference
    if form == "address":
        address = _parse_address(reference)
        repo = next(
            (name for name in PINNED_COMMITS if name.lower() == address.repo.lower()),
            address.repo,
        )
        kind = "datasets/" if address.dataset else ""
        return f"hf://{kind}{repo}@{address.commit}/{address.path}"
    return _resolve(reference, base_dir=base_dir, commits=commits).reference


def _fetch(
    reference: str,
    *,
    base_dir: Path | None = None,
    commits: Mapping[str, str | None] | None = None,
) -> _Save:
    """Resolve a released reference and return its checked local model file.

    Parameters
    ----------
    reference : str
        A typed name, an ``hf://`` address or a released-layout folder.
    base_dir : Path or None, default=None
        Folder for relative local paths; None means the current directory.
    commits : mapping or None, default=None
        Commits recorded by a saved run, used first by typed names.

    Returns
    -------
    _Save
        The saved form, step, SHA-256 and local model file, whose bytes
        match its ``steps.csv`` SHA-256.

    Raises
    ------
    ValueError
        A resolution rule fails, a file is missing, the file's SHA-256 differs
        from ``steps.csv``, or reference is a dataset folder address (see
        _dataset_folder_error).

    Notes
    -----
    Downloads into HF's standard cache only when the file is not cached.
    """
    save = _resolve(reference, base_dir=base_dir, commits=commits)
    if save.reference.startswith("hf://"):
        address = _parse_address(save.reference)
        if address.dataset:
            raise _dataset_folder_error(save.reference)
        path = _hub_file(
            address.repo, address.path, address.commit, reference=reference
        )
    else:
        path = Path(save.reference)
        if not path.is_file():
            raise ValueError(f"{reference!r}: the chosen file {path} is missing")
    if _sha256(path) != save.sha256:
        raise ValueError(
            f"{reference!r}: {path} does not match its SHA-256 in steps.csv; "
            "remove it and download it again"
        )
    return _Save(save.reference, save.step, save.sha256, path)


def _strings(value: object) -> Iterator[str]:
    """Yield every string inside nested JSON-like dicts, lists and tuples."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in cast(Mapping[object, object], value).items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in cast(list[object], value):
            yield from _strings(item)


def _recorded_commits(record: object) -> dict[str, str | None]:
    """Collect the model-repo commits a saved record already uses.

    Parameters
    ----------
    record : object
        A saved description or settings mapping (nested dicts and lists).

    Returns
    -------
    dict of str to (str or None)
        Repo id (as saved) to its commit, for every ``hf://`` model address
        found in record. A repo recorded at more than one commit maps to None,
        so a typed name for it is refused rather than guessed. Dataset
        addresses are left out, because typed names name model repos only.

    Notes
    -----
    Reads only the given object; no file or network access.
    """
    commits: dict[str, str | None] = {}
    for text in _strings(record):
        if not text.startswith("hf://"):
            continue
        try:
            address = _parse_address(text)
        except ValueError:
            continue
        if address.dataset:
            continue
        key = next(
            (name for name in commits if name.lower() == address.repo.lower()),
            address.repo,
        )
        if key in commits and commits[key] != address.commit:
            commits[key] = None
        elif key not in commits:
            commits[key] = address.commit
    return commits


def _identity(reference: str, *, base_dir: Path | None = None) -> dict[str, str]:
    """Describe a released model by content only, so it can move between forms.

    Returns ``{"kind": "portable_model", "sha256": <file SHA-256>}``: the same
    record for a typed name, its ``hf://`` address, its file in a local
    released-layout folder and the same file given directly as a path. Only
    small index files may be downloaded; the model file is not. A dataset
    folder address raises ValueError (see _dataset_folder_error).
    """
    save = _resolve(reference, base_dir=base_dir)
    if save.sha256 is None:
        raise _dataset_folder_error(reference)
    return {"kind": "portable_model", "sha256": save.sha256}


def _dataset_folder_error(reference: str) -> ValueError:
    """Explain how to use an original actor folder from a released dataset.

    Folder loaders refuse symbolic links, and HF's standard cache stores files
    as links on Linux and macOS, so such a folder is used from an ordinary
    local copy instead.
    """
    address = _parse_address(reference)
    return ValueError(
        f"{reference!r} is an original actor folder in a released dataset. "
        "Download it to a local folder and use that folder: hf download "
        f"{address.repo} --repo-type dataset --revision {address.commit} "
        f"--include '{address.path}/*' --local-dir <folder>"
    )


def _released_training_opponents(destination: Path) -> tuple[str, Path]:
    """Download the released training-opponent bundle into a chosen folder.

    Parameters
    ----------
    destination : Path
        Folder that receives ordinary files (not links), for example an
        experiment's ``opponents`` folder. Existing identical files are reused.

    Returns
    -------
    tuple of (str, Path)
        The bundle's exact address
        ``hf://datasets/MARL-BattleGrounds/tdm-season-0-results@<commit>/training-opponents``
        and the local bundle folder ``destination/training-opponents``.

    Raises
    ------
    ValueError
        The dataset cannot be resolved or Hugging Face cannot be reached and
        the files are not already there; the message names opponent_bundle.

    Notes
    -----
    Uses the pinned commit of RESULTS_DATASET, or its latest commit with a
    warning. Writes HF's small download records under ``destination/.cache``.
    """
    hub = _hub()
    try:
        repo, commit = _commit_for(
            RESULTS_DATASET, None, dataset=True, reference=RESULTS_DATASET
        )
        hub.snapshot_download(
            repo,
            repo_type="dataset",
            revision=commit,
            allow_patterns=[f"{TRAINING_OPPONENTS}/*"],
            local_dir=destination,
            library_name=_LIBRARY,
        )
    except (
        ValueError,
        hub.errors.LocalEntryNotFoundError,
        hub.errors.OfflineModeIsEnabled,
    ) as error:
        raise ValueError(
            "The released training opponents could not be found or downloaded "
            f"({error}); give opponent_bundle=/path/opponents instead"
        ) from error
    return f"hf://datasets/{repo}@{commit}/{TRAINING_OPPONENTS}", (
        destination / TRAINING_OPPONENTS
    )


def _training_records(
    repo: str, *, commits: Mapping[str, str | None] | None = None
) -> tuple[str, list[Path]]:
    """Fetch a released repo's training records: only what curves read.

    Parameters
    ----------
    repo : str
        Model repo id, such as ``"MARL-BattleGrounds/tdm-season-0-qmix"``,
        with no ``@`` suffix.
    commits : mapping or None, default=None
        Commits to use first, as for typed names.

    Returns
    -------
    tuple of (str, list of Path)
        The address ``hf://<repo>@<commit>/training-records`` and each run
        record folder under it in HF's standard cache, in name order. Only
        ``run_details.json``, ``status.json`` and ``validation_results.json``
        are downloaded, not the large update logs.

    Raises
    ------
    ValueError
        repo is not a repo id, cannot be resolved, is offline without a
        cached copy, or has no training records at that commit.
    """
    split = _split_name(repo)
    if split is None or split[1] is not None:
        raise ValueError(
            f"{repo!r} must be a released repo id such as MARL-BattleGrounds/<repo>"
        )
    name, commit = _commit_for(split[0], commits, reference=repo)
    hub = _hub()
    try:
        root = Path(
            hub.snapshot_download(
                name,
                revision=commit,
                allow_patterns=[f"training-records/*/{file}" for file in _RECORD_FILES],
                library_name=_LIBRARY,
            )
        )
    except (
        hub.errors.LocalEntryNotFoundError,
        hub.errors.OfflineModeIsEnabled,
    ) as error:
        raise ValueError(
            f"The training records of {name} are not cached and Hugging Face "
            "cannot be reached"
        ) from error
    records = root / "training-records"
    folders = (
        sorted(path for path in records.iterdir() if path.is_dir())
        if records.is_dir()
        else []
    )
    if not folders:
        raise ValueError(f"{name}@{commit} has no training records")
    return f"hf://{name}@{commit}/training-records", folders


def download_models(*names: str) -> list[Path]:
    """Download released models by name so later runs work offline.

    Parameters
    ----------
    *names : str
        Released names, each as typed anywhere MARL-BGs takes a model: a typed
        name such as ``"MARL-BattleGrounds/tdm-season-0-qmix@100M"``, an exact
        ``hf://`` file address, or a local released-layout folder. No names
        means nothing to do.

    Returns
    -------
    list of Path
        The checked local file of each name, in the order given. Files live in
        Hugging Face's standard cache, which ``HF_HOME`` and ``HF_HUB_CACHE``
        choose; ``hf cache`` lists and removes them.

    Raises
    ------
    ValueError
        A name is not a released name, its step does not exist (the message
        lists the nearest saves), a download fails, a file's SHA-256 differs
        from the repo's ``steps.csv``, or the name is a dataset folder address
        (the message gives the ``hf download`` command for a local copy).
    TypeError
        A name is not a string.

    Warns
    -----
    UserWarning
        A repo is not pinned by this package; its latest commit is used.

    Notes
    -----
    Each name downloads its repo's small ``steps.csv`` and ``config.json`` and
    one model file, unless they are already cached. After this, compute nodes
    can run with ``HF_HUB_OFFLINE=1``.

    Examples
    --------
    >>> import marl_battlegrounds as marl_bgs
    >>> marl_bgs.download_models(  # doctest: +SKIP
    ...     "MARL-BattleGrounds/tdm-season-0-qmix@final",
    ...     "MARL-BattleGrounds/tdm-season-0-recurrent-ippo@100M",
    ... )
    """
    paths: list[Path] = []
    for name in names:
        if not isinstance(cast(object, name), str):
            raise TypeError("Released model names must be strings")
        save = _fetch(name)
        assert save.path is not None
        paths.append(save.path)
    return paths
