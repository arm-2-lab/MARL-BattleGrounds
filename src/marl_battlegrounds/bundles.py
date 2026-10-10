"""Copy declared research files into checked, portable ZIP bundles.

export_bundle creates a local archive and download record. prepare_bundle reads
that record, verifies local or explicitly downloaded bytes, then safely unpacks
into a chosen folder. Neither function loads a model, runs an experiment or
publishes files. The existing tournament transfer owner handles HTTP downloads.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlsplit

from marl_battlegrounds.evaluation.tournament_assets import (
    _download_file,  # pyright: ignore[reportPrivateUsage]
    _file_stamp,  # pyright: ignore[reportPrivateUsage]
    _hash_file,  # pyright: ignore[reportPrivateUsage]
    _sync_directory,  # pyright: ignore[reportPrivateUsage]
)

Record = dict[str, Any]
_BUFFER = 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _relative(value: object) -> str:
    """Require one portable relative file path, without traversal or drive syntax."""
    if not isinstance(value, str) or not value:
        raise ValueError("Bundle paths must be nonempty relative file names")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in (".", "..") for part in path.parts)
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or value.endswith("/")
        or value == "."
    ):
        raise ValueError(f"Unsafe bundle path: {value!r}")
    if (
        ".git" in path.parts
        or "__pycache__" in path.parts
        or (
            path.name.startswith("milestone")
            and path.suffix == ".md"
            and any(
                path.parts[index : index + 2] == ("docs", "dev")
                for index in range(len(path.parts) - 2)
            )
        )
    ):
        raise ValueError(f"Keep private/source-history files outside bundles: {value}")
    return value


def _path(value: str | Path) -> Path:
    """Resolve a caller path while rejecting symlinks in every existing component."""
    path = Path(os.path.abspath(Path(value).expanduser()))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Bundle path follows a symbolic link: {path}")
    return path


def _files(root: Path) -> dict[str, Path]:
    """List regular input files in sorted order; refuse private or special entries."""
    if not root.is_dir():
        raise ValueError(f"Bundle source must be a directory: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        name = _relative(path.relative_to(root).as_posix())
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"Bundle contains a link or special file: {path}")
        if path.is_file():
            result[name] = path
    if not result:
        raise ValueError("A bundle must contain at least one file")
    return result


def _identity(value: object, label: str) -> Record:
    """Check one exact byte count and SHA-256 pair without accessing payloads."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} needs size_bytes and sha256")
    row = cast(Record, value)
    if (
        set(row) != {"size_bytes", "sha256"}
        or type(row["size_bytes"]) is not int
        or row["size_bytes"] < 0
        or not isinstance(row["sha256"], str)
        or _DIGEST.fullmatch(row["sha256"]) is None
    ):
        raise ValueError(f"{label} has an invalid byte count or SHA-256")
    return row


def _metadata(value: object) -> Record:
    """Check explicit source, environment, license and citation facts."""
    if not isinstance(value, Mapping):
        raise ValueError("Bundle metadata must be a mapping")
    result = cast(
        Record,
        json.loads(json.dumps(dict(cast(Mapping[str, Any], value)), allow_nan=False)),
    )
    if set(result) != {"source", "environment", "license", "citation"}:
        raise ValueError(
            "Bundle metadata needs source, environment, license and citation"
        )
    for key in ("license", "citation"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"Bundle {key} must be nonempty text")
    for key in ("source", "environment"):
        if not isinstance(result[key], dict) or not result[key]:
            raise ValueError(f"Bundle {key} must be a nonempty evidence mapping")
    return result


def _checked_record(path: Path) -> Record:
    """Read finite JSON and check all declared relative file identities."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> Record:
        """Reject repeated JSON keys instead of silently choosing the last value."""
        result: Record = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Repeated bundle record key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=object_pairs)
    if not isinstance(value, dict):
        raise ValueError("Bundle download record must be an object")
    row = cast(Record, value)
    if (
        type(row.get("schema_version")) is not int
        or row["schema_version"] != 1
        or set(row) != {"schema_version", "archive", "files", "metadata"}
    ):
        raise ValueError("Unsupported bundle download record")
    archive = row["archive"]
    if not isinstance(archive, dict):
        raise ValueError("Bundle archive must be an object")
    archive = cast(Record, archive)
    if not {"path", "size_bytes", "sha256"} <= archive.keys() or archive.keys() - {
        "path",
        "url",
        "size_bytes",
        "sha256",
    }:
        raise ValueError(
            "Bundle archive needs path, size_bytes, sha256 and optional url"
        )
    _relative(archive["path"])
    _identity({key: archive[key] for key in ("size_bytes", "sha256")}, "Archive")
    if "url" in archive and (not isinstance(archive["url"], str) or not archive["url"]):
        raise ValueError("Bundle URL must be nonempty text")
    if "url" in archive and urlsplit(archive["url"]).scheme not in ("http", "https"):
        raise ValueError("Bundle URL must use HTTP or HTTPS")
    contents = row["files"]
    if not isinstance(contents, dict) or not contents:
        raise ValueError("Bundle record needs a nonempty file inventory")
    for name, identity in cast(Record, contents).items():
        _relative(name)
        _identity(identity, name)
    row["metadata"] = _metadata(row["metadata"])
    return row


def _matches(root: Path, inventory: Record) -> bool:
    """Compare an existing destination's complete file inventory without changing it."""
    existing = _files(root)
    if existing.keys() != inventory.keys():
        return False
    return all(
        _hash_file(path) == (inventory[name]["sha256"], inventory[name]["size_bytes"])
        for name, path in existing.items()
    )


def export_bundle(
    source_dir: str | Path,
    *,
    output_dir: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Write a portable ZIP plus an exact download.json record without publishing.

    Parameters
    ----------
    source_dir : path
        Directory of already prepared models, results or source files. Include
        required model provenance and license notices. Reject links, special
        files, Git history, bytecode caches and private milestone documents.
        This does not anonymize arbitrary contents or choose which models are best.
    output_dir : path
        Exact new directory, separate from the input tree. Receives bundle.zip
        and download.json. An occupied destination is refused without changes.
    metadata : mapping
        Exactly source, environment, license and citation. The first two are
        nonempty evidence mappings; the last two are nonempty text. Use the
        actual compatible source and dependency records, not invented versions.
        These documentary facts are retained, not independently certified here.

    Returns
    -------
    Path
        Absolute download.json path. It records every relative file's SHA-256
        and byte count, plus the archive's identity and local path. An optional
        archive.url may be added later for explicit HTTP(S) download.

    Raises
    ------
    ValueError, OSError
        Paths conflict, files change during export, metadata is invalid or I/O
        fails. A failed unpublished archive is removed. A published complete
        bundle survives a later sync failure. Source files are never edited.

    Notes
    -----
    Streams files with bounded buffers and low-cost ZIP compression. No model,
    array, numerical backend or network is loaded. The caller serializes writers.
    Exact hashes prove bytes, not scientific quality or anonymous-review compliance.
    """
    source, target = _path(source_dir), _path(output_dir)
    facts = _metadata(metadata)
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("Bundle output must be separate from its source tree")
    if target.exists():
        raise FileExistsError(f"Bundle destination already exists: {target}")
    inputs = _files(source)
    stamps = {name: _file_stamp(path) for name, path in inputs.items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bundle-", dir=target.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        archive = staged / "bundle.zip"
        inventory: Record = {}
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as writer:
            for name, path in inputs.items():
                digest, size = sha256(), 0
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                # Python 3.13 renamed ZipInfo's level from _compresslevel.
                if sys.version_info >= (3, 13):
                    info.compress_level = 1
                else:
                    info._compresslevel = 1  # pyright: ignore[reportAttributeAccessIssue]
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                with (
                    path.open("rb") as stream,
                    writer.open(info, "w", force_zip64=True) as output,
                ):
                    while chunk := stream.read(_BUFFER):
                        size += len(chunk)
                        digest.update(chunk)
                        output.write(chunk)
                inventory[name] = {"sha256": digest.hexdigest(), "size_bytes": size}
        if inputs != _files(source) or any(
            _file_stamp(path) != stamps[name] for name, path in inputs.items()
        ):
            raise ValueError("Source files changed while writing the bundle")
        digest, size = _hash_file(archive)
        record = {
            "schema_version": 1,
            "archive": {"path": "bundle.zip", "sha256": digest, "size_bytes": size},
            "files": inventory,
            "metadata": facts,
        }
        (staged / "download.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        for path in (archive, staged / "download.json"):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(staged)
        staged.rename(target)
        _sync_directory(target.parent)
    return target / "download.json"


def prepare_bundle(
    record: str | Path,
    *,
    output_dir: str | Path,
    download: bool = False,
) -> Path:
    """Verify and unpack a declared local or explicitly downloaded bundle.

    Parameters
    ----------
    record : path
        download.json from export_bundle, with an optional HTTP(S) archive.url.
        archive.path is relative to this record. Local corrupt files raise;
        they are never silently replaced by a download.
    output_dir : path
        Exact destination for the archive's relative contents. A new folder is
        published only after every member passes. Reuse an identical complete
        directory; reject conflicts, symlinks and overlap with the record/archive.
    download : bool, default=False
        True permits one bounded HTTP(S) transfer only when the local archive is
        missing. False needs local bytes. Importing this module never downloads.

    Returns
    -------
    Path
        Absolute verified output folder, ready for the existing model/result APIs.
        No model or experiment is loaded automatically.

    Raises
    ------
    ValueError, OSError, zipfile.BadZipFile
        The record, sizes, hashes or paths are invalid; the archive has links,
        traversal, duplicate/undeclared members or unsupported file types; outputs
        conflict; or I/O fails. Failed unpublished extraction is cleaned up.
        The declared inventory bounds extracted bytes even for compressed files.
    """
    if type(download) is not bool:
        raise TypeError("download must be true or false")
    declaration, target = _path(record), _path(output_dir)
    saved = _checked_record(declaration)
    archive = _path(declaration.parent / saved["archive"]["path"])
    if (
        target in (declaration, declaration.parent, archive)
        or declaration in target.parents
        or archive in target.parents
        or target in declaration.parents
        or target in archive.parents
    ):
        raise ValueError("Bundle output must stay separate from its download inputs")
    if target.exists():
        if target.is_dir() and _matches(target, saved["files"]):
            return target
        raise ValueError(f"Bundle destination conflicts: {target}")
    if not archive.exists() and not (download and saved["archive"].get("url")):
        raise FileNotFoundError(
            f"Bundle archive is missing: {archive}; "
            "supply it or set download=True with a URL"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bundle-", dir=target.parent) as temporary:
        staging = Path(temporary)
        if not archive.exists():
            archive = staging / "download.zip"
            _download_file(
                saved["archive"]["url"],
                archive,
                size_bytes=saved["archive"]["size_bytes"],
                sha256_digest=saved["archive"]["sha256"],
            )
        elif _hash_file(archive) != (
            saved["archive"]["sha256"],
            saved["archive"]["size_bytes"],
        ):
            raise ValueError("Bundle archive has the wrong size or hash")
        unpacked = staging / "contents"
        unpacked.mkdir()
        with zipfile.ZipFile(archive) as reader:
            entries = reader.infolist()
            names = [_relative(entry.filename) for entry in entries]
            if len(set(names)) != len(names) or set(names) != set(saved["files"]):
                raise ValueError("ZIP entries differ from the declared inventory")
            for entry, name in zip(entries, names, strict=True):
                identity = saved["files"][name]
                kind = stat.S_IFMT(entry.external_attr >> 16)
                if (
                    kind not in (0, stat.S_IFREG)
                    or entry.is_dir()
                    or entry.file_size != identity["size_bytes"]
                ):
                    raise ValueError(f"ZIP entry has an invalid type or size: {name}")
                destination = unpacked / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest, size = sha256(), 0
                with reader.open(entry) as stream, destination.open("xb") as output:
                    while chunk := stream.read(
                        min(_BUFFER, identity["size_bytes"] - size + 1)
                    ):
                        size += len(chunk)
                        if size > identity["size_bytes"]:
                            raise ValueError(
                                f"ZIP entry exceeds its declared size: {name}"
                            )
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if (
                    size != identity["size_bytes"]
                    or digest.hexdigest() != identity["sha256"]
                ):
                    raise ValueError(f"ZIP entry has the wrong size or hash: {name}")
        for directory in sorted(
            (path for path in unpacked.rglob("*") if path.is_dir()), reverse=True
        ):
            _sync_directory(directory)
        _sync_directory(unpacked)
        unpacked.rename(target)
        _sync_directory(target.parent)
    return target
