"""Open a link in the user's browser, including from WSL2 on Windows.

The BattleClient, the Replay Viewer and ``entry submit`` all open their link
through ``open_in_browser``, so the rule for finding a browser lives here.

Python's ``webbrowser`` cannot reach Windows from inside WSL2: it calls a Linux
helper such as ``gio``, which prints "Operation not supported" and opens
nothing. Inside WSL, ``open_in_browser`` therefore asks Windows to open the
link with PowerShell's ``Start-Process``, which hands the whole link, including
any ``#token=...`` part, to the Windows default browser. Everywhere else, and
inside WSL when PowerShell cannot be found, it uses ``webbrowser.open``.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import webbrowser
from pathlib import Path

# Where Windows keeps PowerShell, as WSL mounts drive C: by default. A login
# shell started with sudo -i has no Windows folders on PATH.
_POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")


def _in_wsl() -> bool:
    """Return whether this Python runs inside WSL (version 1 or 2).

    WSL kernels name Microsoft in their release string, for example
    ``6.6.87.2-microsoft-standard-WSL2``. Docker Desktop's Linux containers
    run on such a kernel too; they find no PowerShell, so ``open_in_browser``
    falls back to ``webbrowser`` there.
    """
    return "microsoft" in platform.uname().release.lower()


def _windows_powershell() -> str | None:
    """Return a PowerShell program WSL can start, or None when there is none."""
    found = shutil.which("powershell.exe")
    if found is not None:
        return found
    return str(_POWERSHELL) if _POWERSHELL.is_file() else None


def open_in_browser(url: str) -> bool:
    """Open ``url`` in the user's default browser.

    Parameters
    ----------
    url : str
        The full link to open, for example the BattleClient's
        ``http://127.0.0.1:PORT/#token=...`` address. It is passed on whole.

    Returns
    -------
    bool
        True when the browser was asked to open the link. Inside WSL with
        PowerShell, that means PowerShell's ``Start-Process`` exited with
        status 0. Elsewhere it is what ``webbrowser.open`` returns. False
        means the caller should tell the user to open the link by hand.

    Raises
    ------
    Exception
        Whatever ``webbrowser.open`` raises when no browser can be started.
        Starting PowerShell never raises: a failure returns False.

    Notes
    -----
    Inside WSL this starts ``powershell.exe`` (about a second) and waits for
    it; the link sits in single quotes, with any single quote doubled, so
    PowerShell reads it as plain text. Nothing else on the machine changes.
    """
    if _in_wsl():
        powershell = _windows_powershell()
        if powershell is not None:
            quoted = "'" + url.replace("'", "''") + "'"
            try:
                finished = subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        f"Start-Process {quoted}",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except OSError:
                return False
            return finished.returncode == 0
    return webbrowser.open(url)
