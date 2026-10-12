"""Check how ``_browser.open_in_browser`` opens a link.

Inside WSL (a kernel release naming Microsoft) with PowerShell found, either on
``PATH`` or at Windows' own folder under ``/mnt/c``, it runs ``powershell.exe
-NoProfile -NonInteractive -Command Start-Process '<link>'`` with the whole link
(``#token=...`` included, any single quote doubled), never calls
``webbrowser.open``, and returns whether PowerShell exited with status 0; a
PowerShell that cannot start returns False. Outside WSL, or inside WSL with no
PowerShell, it returns what ``webbrowser.open`` returns and starts nothing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from marl_battlegrounds import _browser

_LINK = "http://127.0.0.1:8765/#token=ab-C_9"
_WSL = "6.6.87.2-microsoft-standard-WSL2"


def _set_up(
    monkeypatch: pytest.MonkeyPatch,
    *,
    release: str,
    on_path: str | None,
    exit_status: int = 0,
    start_error: bool = False,
) -> tuple[list[list[str]], list[str]]:
    started: list[list[str]] = []
    opened: list[str] = []

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if start_error:
            raise OSError("Windows programs are turned off")
        started.append(command)
        return SimpleNamespace(returncode=exit_status)

    def browser_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(
        _browser.platform, "uname", lambda: SimpleNamespace(release=release)
    )

    def which(_name: str) -> str | None:
        return on_path

    monkeypatch.setattr(_browser.shutil, "which", which)
    monkeypatch.setattr(_browser.subprocess, "run", run)
    monkeypatch.setattr(_browser.webbrowser, "open", browser_open)
    return started, opened


@pytest.mark.parametrize(("exit_status", "expected"), [(0, True), (1, False)])
def test_wsl_asks_windows_to_open_the_whole_link(
    monkeypatch: pytest.MonkeyPatch, exit_status: int, expected: bool
) -> None:
    started, opened = _set_up(
        monkeypatch,
        release=_WSL,
        on_path="/win/powershell.exe",
        exit_status=exit_status,
    )
    assert _browser.open_in_browser(_LINK) is expected
    assert started == [
        [
            "/win/powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"Start-Process '{_LINK}'",
        ]
    ]
    assert opened == []


def test_a_single_quote_in_the_link_is_doubled(monkeypatch: pytest.MonkeyPatch) -> None:
    started, _ = _set_up(monkeypatch, release=_WSL, on_path="powershell.exe")
    assert _browser.open_in_browser("http://x/?q=it's")
    assert started[0][-1] == "Start-Process 'http://x/?q=it''s'"


def test_wsl_finds_powershell_in_its_windows_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    powershell = tmp_path / "powershell.exe"
    powershell.write_text("", encoding="utf-8")
    monkeypatch.setattr(_browser, "_POWERSHELL", powershell)
    started, opened = _set_up(monkeypatch, release=_WSL, on_path=None)
    assert _browser.open_in_browser(_LINK)
    assert started[0][0] == str(powershell)
    assert opened == []


def test_wsl_without_powershell_uses_webbrowser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_browser, "_POWERSHELL", tmp_path / "missing.exe")
    started, opened = _set_up(monkeypatch, release=_WSL, on_path=None)
    assert _browser.open_in_browser(_LINK)
    assert started == []
    assert opened == [_LINK]


def test_other_systems_use_webbrowser(monkeypatch: pytest.MonkeyPatch) -> None:
    started, opened = _set_up(
        monkeypatch, release="6.8.0-45-generic", on_path="/win/powershell.exe"
    )
    assert _browser.open_in_browser(_LINK)
    assert started == []
    assert opened == [_LINK]


def test_powershell_that_cannot_start_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, opened = _set_up(
        monkeypatch, release=_WSL, on_path="/win/powershell.exe", start_error=True
    )
    assert _browser.open_in_browser(_LINK) is False
    assert opened == []
