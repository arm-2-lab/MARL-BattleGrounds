"""Check that public launchers find the browser page of a running source checkout.

Contracts: ``checkout_browser_files`` returns None for an installed package (its
``web`` folder exists) and for any resource that is not a filesystem path. For a
missing packaged page it returns ``web/visual_debugger`` only when the folder three
levels above ``src/marl_battlegrounds/viewer`` holds both ``pyproject.toml`` and
``web/visual_debugger/index.html``; any missing piece returns None, so the caller
keeps its existing package lookup and errors. It reads existence only and writes
nothing. ``launch_replay`` and the BattleClient's ``_launch_setup`` serve that
checkout page without extracting package resources, and an explicit asset root
still wins over both.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from pathlib import Path

import pytest

from marl_battlegrounds.viewer.launch import checkout_browser_files


def _package_files(packaged: Path) -> Callable[[str], Path]:
    def files(_name: str) -> Path:
        return packaged.parent

    return files


def _no_binding(_service: object) -> None:
    return None


def _checkout(root: Path, *, pyproject: bool = True, page: bool = True) -> Path:
    viewer = root / "src" / "marl_battlegrounds" / "viewer"
    viewer.mkdir(parents=True)
    if pyproject:
        (root / "pyproject.toml").write_text("[project]\n")
    if page:
        (root / "web" / "visual_debugger").mkdir(parents=True)
        (root / "web" / "visual_debugger" / "index.html").write_text("<!doctype html>")
    return viewer / "web"


def test_installed_package_page_is_never_replaced(tmp_path: Path) -> None:
    packaged = _checkout(tmp_path)
    packaged.mkdir()
    assert checkout_browser_files(packaged) is None


def test_non_filesystem_resources_are_left_to_the_package_lookup() -> None:
    assert checkout_browser_files("marl_battlegrounds/viewer/web") is None


def test_checkout_page_is_found_beside_the_running_code(tmp_path: Path) -> None:
    packaged = _checkout(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert checkout_browser_files(packaged) == tmp_path / "web" / "visual_debugger"
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("missing", ["pyproject", "page"])
def test_incomplete_checkouts_keep_the_package_lookup(
    tmp_path: Path, missing: str
) -> None:
    packaged = _checkout(
        tmp_path, pyproject=missing != "pyproject", page=missing != "page"
    )
    assert checkout_browser_files(packaged) is None


def test_replay_viewer_serves_the_checkout_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = importlib.import_module("marl_battlegrounds.viewer.launch")
    server = importlib.import_module("marl_battlegrounds.viewer.server")
    replay_service = importlib.import_module("marl_battlegrounds.viewer.replay_service")
    options_module = importlib.import_module("marl_battlegrounds.viewer.options")
    packaged = _checkout(tmp_path)
    served: list[Path] = []

    class Service:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def prepare_agent_pov(self) -> bool:
            return True

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("A checkout page needs no package extraction.")

    def serve(_service: object, **kwargs: object) -> int:
        root = kwargs["asset_root"]
        assert isinstance(root, Path)
        served.append(root)
        return 5

    monkeypatch.setattr(launch, "files", _package_files(packaged))
    monkeypatch.setattr(launch, "as_file", forbidden)
    monkeypatch.setattr(launch, "replay_http_binding", _no_binding)
    monkeypatch.setattr(replay_service, "ReplayViewerService", Service)
    monkeypatch.setattr(server, "serve_browser_debugger", serve)
    options = options_module.resolve_playback_options(argparse.Namespace(no_open=True))
    bundle = object()
    assert launch.launch_replay(None, options=options, loaded_bundle=bundle) == 5
    assert served == [tmp_path / "web" / "visual_debugger"]


@pytest.mark.parametrize("explicit", [False, True])
def test_battle_client_serves_the_checkout_page_unless_given_a_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    from tests.battle_client.test_battle_client_launch import (
        _stub_launch_session,  # pyright: ignore[reportPrivateUsage]
    )

    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    setup, events = _stub_launch_session(monkeypatch)
    packaged = _checkout(tmp_path / "repo")
    explicit_root = tmp_path / "explicit"
    served: list[object] = []

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("A checkout page needs no package extraction.")

    def serve(_service: object, **kwargs: object) -> int:
        served.append(kwargs["asset_root"])
        return 9

    monkeypatch.setattr(launch, "files", _package_files(packaged))
    monkeypatch.setattr(launch, "as_file", forbidden)
    monkeypatch.setattr(launch, "serve_browser_debugger", serve)
    result = launch._launch_setup(  # pyright: ignore[reportPrivateUsage]
        setup,
        port=0,
        open_browser=False,
        output_dir=tmp_path,
        capture_replay=False,
        asset_root=explicit_root if explicit else None,
    )
    assert result == 9
    assert served == [
        explicit_root if explicit else tmp_path / "repo" / "web" / "visual_debugger"
    ]
    assert events == ["warm", "service", "close"]
