"""Check complete portable bundle export, local/HTTP preparation and failure safety.

Use small real files and a local server. No model, provider, numerical backend or
public download runs. Hashes, bounded extraction, path safety, relocation and
published-state preservation are the contracts; scientific quality is not inferred.
"""

# pyright: reportPrivateUsage=false
import json
import shutil
import stat
import subprocess
import sys
import threading
import zipfile
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import bundles

METADATA = {
    "source": {"kind": "Software Fixture", "sha256": "a" * 64},
    "environment": {"kind": "Host Files Only"},
    "license": "Fixture Permission",
    "citation": "Synthetic software fixture; no research result",
}


def _source(root: Path) -> Path:
    root.mkdir()
    (root / "models").mkdir()
    (root / "models/weights.bin").write_bytes(bytes(range(256)) * 1024)
    (root / "LICENSE").write_text("Fixture Permission\n")
    return root


def _archive(root: Path, names: list[str], *, mode: int = stat.S_IFREG) -> Path:
    root.mkdir()
    payload = b"checked bytes"
    archive = root / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        for name in names:
            info = zipfile.ZipInfo(name)
            info.external_attr = (mode | 0o644) << 16
            writer.writestr(info, payload)
    raw = archive.read_bytes()
    record = {
        "schema_version": 1,
        "archive": {
            "path": "bundle.zip",
            "size_bytes": len(raw),
            "sha256": sha256(raw).hexdigest(),
        },
        "files": {
            name: {"size_bytes": len(payload), "sha256": sha256(payload).hexdigest()}
            for name in names
        },
        "metadata": METADATA,
    }
    target = root / "download.json"
    target.write_text(json.dumps(record))
    return target


def test_export_relocate_prepare_and_reuse(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    originals = {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    record = marl_bgs.export_bundle(
        source, output_dir=tmp_path / "release", metadata=METADATA
    )
    moved = tmp_path / "moved"
    record.parent.rename(moved)
    shutil.rmtree(source)
    output = tmp_path / "ready"
    assert marl_bgs.prepare_bundle(moved / "download.json", output_dir=output) == output
    assert {
        str(p.relative_to(output)): p.read_bytes()
        for p in output.rglob("*")
        if p.is_file()
    } == originals
    stamps = {
        str(p.relative_to(output)): p.stat().st_mtime_ns for p in output.rglob("*")
    }
    assert marl_bgs.prepare_bundle(moved / "download.json", output_dir=output) == output
    assert stamps == {
        str(p.relative_to(output)): p.stat().st_mtime_ns for p in output.rglob("*")
    }
    (output / "LICENSE").write_text("Different bytes")
    with pytest.raises(ValueError, match="conflicts"):
        marl_bgs.prepare_bundle(moved / "download.json", output_dir=output)
    assert (output / "LICENSE").read_text() == "Different bytes"


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a/../escape", "a\\escape", "C:/drive"]
)
def test_unsafe_member_paths_never_publish(tmp_path: Path, name: str) -> None:
    record = _archive(tmp_path / "release", [name])
    with pytest.raises(ValueError, match="Unsafe bundle path"):
        marl_bgs.prepare_bundle(record, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "problem", ["duplicate", "symlink", "fifo", "hash", "missing", "extra", "size"]
)
def test_bad_archives_never_publish(tmp_path: Path, problem: str) -> None:
    names = ["data"] * (2 if problem == "duplicate" else 1)
    mode = (
        stat.S_IFLNK
        if problem == "symlink"
        else stat.S_IFIFO
        if problem == "fifo"
        else stat.S_IFREG
    )
    record = _archive(tmp_path / "release", names, mode=mode)
    value = json.loads(record.read_text())
    if problem == "hash":
        value["files"]["data"]["sha256"] = "0" * 64
    elif problem == "size":
        value["files"]["data"]["size_bytes"] -= 1
    elif problem == "missing":
        value["files"]["absent"] = value["files"].pop("data")
    elif problem == "extra":
        value["files"]["absent"] = value["files"]["data"]
    record.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        marl_bgs.prepare_bundle(record, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".bundle-*"))


def test_explicit_local_http_download(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    record = marl_bgs.export_bundle(
        source, output_dir=tmp_path / "server", metadata=METADATA
    )
    handler = partial(SimpleHTTPRequestHandler, directory=str(record.parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        value = json.loads(record.read_text())
        value["archive"]["path"] = "not-downloaded.zip"
        value["archive"]["url"] = f"http://127.0.0.1:{server.server_port}/bundle.zip"
        local = tmp_path / "download.json"
        local.write_text(json.dumps(value))
        with pytest.raises(FileNotFoundError):
            marl_bgs.prepare_bundle(local, output_dir=tmp_path / "out")
        assert (
            marl_bgs.prepare_bundle(local, output_dir=tmp_path / "out", download=True)
            == tmp_path / "out"
        )
        assert (tmp_path / "out/models/weights.bin").read_bytes() == (
            source / "models/weights.bin"
        ).read_bytes()
        assert not (tmp_path / "not-downloaded.zip").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("published", [False, True])
def test_export_failure_preserves_publication_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published: bool
) -> None:
    source = _source(tmp_path / "source")
    output = tmp_path / "release"
    original = bundles._sync_directory

    def fail(path: Path) -> None:
        if (path == output.parent) == published:
            raise OSError("Injected sync failure")
        original(path)

    monkeypatch.setattr(bundles, "_sync_directory", fail)
    with pytest.raises(OSError, match="Injected"):
        marl_bgs.export_bundle(source, output_dir=output, metadata=METADATA)
    assert output.exists() == published
    assert not list(tmp_path.glob(".bundle-*"))
    if published:
        monkeypatch.setattr(bundles, "_sync_directory", original)
        assert marl_bgs.prepare_bundle(
            output / "download.json", output_dir=tmp_path / "ready"
        ).is_dir()


def test_export_rejects_symlinks_and_never_overwrites(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    (source / "linked").symlink_to(source / "LICENSE")
    with pytest.raises(ValueError, match="link"):
        marl_bgs.export_bundle(
            source, output_dir=tmp_path / "release", metadata=METADATA
        )
    assert not (tmp_path / "release").exists()
    (source / "linked").unlink()
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "keep").write_text("Keep")
    with pytest.raises(FileExistsError):
        marl_bgs.export_bundle(source, output_dir=target, metadata=METADATA)
    assert (target / "keep").read_text() == "Keep"


def test_invalid_record_does_not_create_output(tmp_path: Path) -> None:
    path = tmp_path / "download.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="Repeated"):
        marl_bgs.prepare_bundle(path, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "name", [".git/config", "__pycache__/code.pyc", "docs/dev/milestone_12.md"]
)
def test_private_files_rejected_on_both_routes(tmp_path: Path, name: str) -> None:
    record = _archive(tmp_path / "release", [name])
    with pytest.raises(ValueError, match="private/source-history"):
        marl_bgs.prepare_bundle(record, output_dir=tmp_path / "new-parent/out")
    assert not (tmp_path / "new-parent").exists()
    source = _source(tmp_path / "source")
    entry = source / name
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("Private")
    with pytest.raises(ValueError, match="private/source-history"):
        marl_bgs.export_bundle(
            source, output_dir=tmp_path / "new-parent/out", metadata=METADATA
        )
    assert not (tmp_path / "new-parent").exists()


def test_bundle_api_is_host_only() -> None:
    script = """
import sys
class Guard:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'jax', 'flax', 'optax', 'flashbax', 'torch'}:
            raise AssertionError(name)
sys.meta_path.insert(0, Guard())
import marl_battlegrounds as marl_bgs
assert callable(marl_bgs.export_bundle)
assert callable(marl_bgs.prepare_bundle)
"""
    subprocess.run([sys.executable, "-B", "-c", script], check=True)


def test_source_change_cleans_unpublished_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path / "source")
    original = bundles._file_stamp
    calls = 0

    def changed(path: Path) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        if calls == 3:
            (source / "LICENSE").write_text("Changed during export")
        return original(path)

    monkeypatch.setattr(bundles, "_file_stamp", changed)
    with pytest.raises(ValueError, match="changed while writing"):
        marl_bgs.export_bundle(
            source, output_dir=tmp_path / "release", metadata=METADATA
        )
    assert not (tmp_path / "release").exists()
    assert not list(tmp_path.glob(".bundle-*"))


@pytest.mark.parametrize("published", [False, True])
def test_unpack_failure_preserves_publication_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published: bool
) -> None:
    record = _archive(tmp_path / "release", ["data"])
    output = tmp_path / "out"
    original = bundles._sync_directory

    def fail(path: Path) -> None:
        if (path == output.parent) == published:
            raise OSError("Injected extraction failure")
        original(path)

    with monkeypatch.context() as patch:
        patch.setattr(bundles, "_sync_directory", fail)
        with pytest.raises(OSError, match="Injected"):
            marl_bgs.prepare_bundle(record, output_dir=output)
    assert output.exists() == published
    assert not list(tmp_path.glob(".bundle-*"))
    assert marl_bgs.prepare_bundle(record, output_dir=output) == output
