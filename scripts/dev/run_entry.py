"""Run leaderboard entries sealed off from the network and this computer's secrets.

Maintainers use this script for the intake test and the official run of each
leaderboard entry. Both are one call of ``marl_battlegrounds.entry.check_entry``,
the same check entrants run at home; this script adds the seal and the limits
around it. Name the card in ``CUDA_VISIBLE_DEVICES``::

    CUDA_VISIBLE_DEVICES=GPU-<uuid> uv run --no-sync python scripts/dev/run_entry.py \\
        intake REPOSITORY COMMIT FOLDER
    CUDA_VISIBLE_DEVICES=GPU-<uuid> uv run --no-sync python scripts/dev/run_entry.py \\
        official REPOSITORY COMMIT FOLDER
    CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/dev/run_entry.py \\
        run CLONE -- COMMAND [ARGUMENT ...]

- ``intake`` checks COMMIT of REPOSITORY with 15 minutes of training, in a
  temporary folder under /tmp that the check removes.
- ``official`` is the same check with 8 hours of training, in
  ``FOLDER/official/run``, which must not exist yet and is kept: the clone is
  ``FOLDER/official/run/entry``, with the run's output and its model cache.
- Both start a systemd user unit and return at once. The unit is
  ``marl-bgs-intake-<short>`` or ``marl-bgs-official-<short>``, where
  ``<short>`` is the commit's first 12 characters. Each kind writes only in its
  own folder, ``FOLDER/intake`` or ``FOLDER/official``: the log
  ``check-<short>.log`` (each start adds to it), and in ``marl-bgs-entry-check/``
  the record ``<commit>.json`` and ``<commit>.bundle``, a copy of the code that
  ran (``<commit>`` is the full SHA). ``systemctl --user status <unit>`` shows
  the unit, and after it ends ``journalctl --user -u <unit>`` says how it ended.
  ``systemctl --user stop <unit>`` ends the check at once: training gets no
  time to save, and no record is written.
- ``run`` runs one command in an entry's clone, sealed and offline, and waits
  for it. The protocol uses it to export the last saved checkpoint when the
  official run saved no choice. Only the clone is writable inside the seal, so
  the command must write its file inside the clone (for example
  ``CLONE/entry.zip``).
- ``intake`` and ``official`` refuse to start until the install's seal is
  built (see ``sandbox``); ``run`` works now.

The seal is ``sandbox``'s prefix, put in front of every program that runs the
entry's own code. Each command runs in a systemd user unit that also sets:

- 48 GB of RAM and no swap for everything in it. The kernel kills a program
  that goes past 48 GB; the check carries on and records the failure.
- A stop 3 hours after the official training time, well above the check's own
  timers (see ``check_entry``), so it never stops a healthy run.
- Only ``PATH``, ``HOME``, ``LANG`` and ``CUDA_VISIBLE_DEVICES`` in the
  environment, so no token or key from the maintainer's shell reaches the
  check or the entry.

A check's log ends with the unit's peak RAM, from its cgroup's ``memory.peak``
(file cache included). systemd-oomd may still kill a unit under heavy memory
pressure before 48 GB (this machine's user manager kills at 50% pressure);
``journalctl -u systemd-oomd`` then names it.

Prerequisites: Linux with bubblewrap (``bwrap``), systemd user units and uv on
``PATH``, and FOLDER inside the home folder: the seal hides the home folder,
and with it every other entry's folder. Run ``loginctl enable-linger`` once,
so that a unit keeps running after a logout.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from marl_battlegrounds import entry

_UNIT_PROPERTIES = (
    "--property=MemoryMax=48000000000",
    "--property=MemorySwapMax=0",
    f"--property=RuntimeMaxSec={entry.OFFICIAL_TRAIN_MINUTES / 60 + 3:g}h",
    "--property=KillSignal=SIGINT",
    "--property=TimeoutStopSec=120",
    "--property=OOMPolicy=continue",
)
_IN_UNIT = "--in-unit"
_NO_INSTALL_SEAL = (
    "The install's seal is not built yet: it needs a network of its own, made "
    "with pasta (sudo apt install passt) and proven on this machine. Nothing "
    "was run."
)


def sandbox(clone: Path, online: bool) -> list[str]:
    """Return the command prefix that seals one of the entry's own programs.

    ``check_entry`` puts it in front of every child that runs the entry's code,
    and ``run`` uses it too. Inside, the program sees the whole file system
    read-only, the card that ``CUDA_VISIBLE_DEVICES`` names (only that card's
    device, so it cannot use another; none when it is unset or empty), and
    ``clone`` writable. This computer's home
    folder, ``/run`` (Docker's socket, D-Bus, the SSH agent and the desktop's
    sockets), ``/media`` and ``/mnt`` (other drives) and the real ``/tmp`` are
    hidden. The program has its own network, processes, shared memory and host
    name, so it cannot reach the internet or this computer's loopback ports and
    abstract sockets (such as the desktop's display), and cannot see or signal
    this computer's other programs. If uv and its Python installs (``uv python
    dir``) live in the home folder, they alone are shown there, read-only: an
    environment that uv built outside the seal links its ``python`` to one of
    them.

    Parameters
    ----------
    clone : Path
        The entry's clone, inside the home folder or ``/tmp``, so that the seal
        hides every other entry's folder. Two folders are made inside it if
        missing: ``.entry-check-home``, which becomes ``HOME``, and
        ``.entry-check-tmp``, which becomes ``/tmp``. So caches and temporary
        files count as the entry's disk, not RAM.
    online : bool
        True for a program that needs the network (step 4's install and
        downloads); False for training, loading, export and play.

    Returns
    -------
    list of str
        ``env --ignore-signal=INT bwrap ... env --default-signal=INT``. Append
        the command. The program keeps the caller's environment, except that
        ``HOME`` and ``TMPDIR`` point inside the clone and ``WANDB_MODE`` is
        ``offline``.

    Raises
    ------
    RuntimeError
        ``online`` is True: the install needs a network of its own, so that it
        reaches the internet but not this computer's desktop or local ports.
        That needs pasta and a proof on this machine, which are not done yet.
        Also raised before anything is made when bubblewrap is missing or
        ``clone`` is outside the home folder and ``/tmp``.

    Notes
    -----
    Signals: bwrap ignores SIGINT and the program gets the default back, so a
    stop signal sent to the program's process group reaches the program, not
    bwrap, and a trainer can save before it ends. There is deliberately no
    ``--die-with-parent``, which kills the program before it handles the
    signal, and no ``--new-session``, which takes the program out of the
    process group. Without a new session the program could write to a
    terminal it shares, so this prefix is for children with no terminal: those
    of ``check_entry`` (each starts a new session) and those inside a systemd
    unit.
    """
    if online:
        raise RuntimeError(_NO_INSTALL_SEAL)
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("The seal needs bubblewrap: sudo apt install bubblewrap")
    clone = clone.resolve()
    real_home = Path.home()
    if not (clone.is_relative_to(real_home) or clone.is_relative_to("/tmp")):
        raise RuntimeError(
            f"{clone} is outside {real_home} and /tmp, where the seal cannot hide "
            "the other entries' folders. Nothing was run."
        )
    home, tmp = clone / ".entry-check-home", clone / ".entry-check-tmp"
    home.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)
    uv = shutil.which("uv")
    pythons = (
        subprocess.run(
            [uv, "python", "dir"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if uv is not None
        else ""
    )
    shown = [
        path
        for path in (uv or "", pythons)
        if path and Path(path).exists() and Path(path).is_relative_to(real_home)
    ]
    return [
        *("env", "--ignore-signal=INT", bwrap),
        *("--ro-bind", "/", "/", "--dev", "/dev"),
        *(arg for device in _gpu_devices() for arg in ("--dev-bind", device, device)),
        *("--proc", "/proc", "--tmpfs", "/run"),
        *(
            arg
            for drives in ("/media", "/mnt")
            if Path(drives).is_dir()
            for arg in ("--tmpfs", drives)
        ),
        *("--tmpfs", str(real_home)),
        *(arg for path in shown for arg in ("--ro-bind", path, path)),
        # /tmp comes before the clone, because the clone may live under /tmp.
        *("--bind", str(tmp), "/tmp", "--bind", str(clone), str(clone)),
        "--unshare-all",
        *("--setenv", "HOME", str(home), "--setenv", "TMPDIR", "/tmp"),
        *("--setenv", "WANDB_MODE", "offline"),
        *("env", "--default-signal=INT"),
    ]


def _gpu_devices() -> list[str]:
    """Return the NVIDIA device files for the cards ``CUDA_VISIBLE_DEVICES`` names.

    Each named card (a UUID or an index) becomes its own ``/dev/nvidia<N>``,
    found with ``nvidia-smi -q -i``, plus the shared control devices that
    exist (``nvidiactl``, ``nvidia-uvm``, ``nvidia-uvm-tools``,
    ``nvidia-modeset``, ``nvidia-caps``). Returns an empty list when the
    variable is unset or empty, or when no card is named, so the seal then
    shows no GPU at all.
    """
    cards = [
        card for card in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if card
    ]
    if not cards:
        return []
    found: list[str] = []
    for card in cards:
        query = ["nvidia-smi", "-q", "-i", card]
        report = subprocess.run(
            query, capture_output=True, text=True, check=False
        ).stdout
        minor = re.search(r"Minor Number\s*:\s*(\d+)", report)
        if minor:
            found.append(f"/dev/nvidia{minor.group(1)}")
    shared = (
        "nvidiactl",
        "nvidia-uvm",
        "nvidia-uvm-tools",
        "nvidia-modeset",
        "nvidia-caps",
    )
    return found + [f"/dev/{name}" for name in shared if Path(f"/dev/{name}").exists()]


def _tool_path() -> str:
    """Return a PATH for a unit: uv's folder, then the usual system folders."""
    uv = shutil.which("uv")
    folders = ([str(Path(uv).parent)] if uv else []) + [
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ]
    return os.pathsep.join(dict.fromkeys(folders))


def _unit_command(
    directory: Path,
    arguments: Sequence[str],
    *,
    unit: str,
    detach: bool,
    log: Path | None,
) -> list[str]:
    """Return the systemd-run command that runs this script again inside a unit.

    Parameters
    ----------
    directory : Path
        The unit's working folder.
    arguments : sequence of str
        The script's own arguments inside the unit (they include ``--in-unit``).
    unit : str
        The unit's name.
    detach : bool
        True returns at once while the unit runs; False waits for it and passes
        its output through (its input is closed).
    log : Path or None
        A file that gets the unit's output, added to its end; None keeps it.

    Returns
    -------
    list of str
        ``systemd-run --user ... env -i PATH=... HOME=... LANG=C.UTF-8
        CUDA_VISIBLE_DEVICES=... python -u run_entry.py ...``. ``env -i``
        starts the unit's program with only those four variables, so nothing
        from the maintainer's shell or the session manager reaches it. An unset
        ``CUDA_VISIBLE_DEVICES`` becomes empty, which hides the GPU.
        ``--expand-environment=no`` stops systemd from rewriting ``$`` in the
        arguments.
    """
    variables = {
        "PATH": _tool_path(),
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    return [
        *("systemd-run", "--user", "--collect", "--quiet", "--expand-environment=no"),
        *_UNIT_PROPERTIES,
        f"--unit={unit}",
        *([] if detach else ["--wait", "--pipe"]),
        *([f"--property=StandardOutput=append:{log}"] if log else []),
        f"--working-directory={directory}",
        *("env", "-i", *(f"{name}={value}" for name, value in variables.items())),
        *(sys.executable, "-u", str(Path(__file__).resolve()), *arguments),
    ]


def _start(kind: str, repository: str, commit: str, folder: Path) -> int:
    """Start the intake test or the official run as a systemd user unit.

    Parameters
    ----------
    kind : {"intake", "official"}
        Which check: 15 minutes of training in a temporary folder, or 8 hours
        kept in ``folder / "official" / "run"``.
    repository, commit : str
        What to check: a repository URL or path (a local path is made absolute),
        and the full commit SHA. ``check_entry`` checks the SHA's form inside
        the unit.
    folder : Path
        The entry's folder, inside the home folder. ``folder / kind`` is made
        if missing and gets the log, the record and, for the official run, the
        run itself (see the top of this file).

    Returns
    -------
    int
        0 once the unit has started, while the check itself still runs; else
        systemd-run's exit status, for example when a unit of the same name is
        still running.

    Raises
    ------
    SystemExit
        No unit starts when ``CUDA_VISIBLE_DEVICES`` is unset or empty (no card
        would be used); when ``folder`` is outside the home folder; when this
        commit already has a record of this kind, or the official run's folder
        exists (move ``folder / kind`` aside to run it again); when lingering is
        off (a logout would stop the unit); or while the install's seal is not
        built (every check would fail at step 4 for a reason of ours; see
        ``sandbox``).
    """
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES to the card the protocol names.")
    folder = folder.resolve()
    if not folder.is_relative_to(Path.home()):
        raise SystemExit(
            f"Put FOLDER inside {Path.home()}: the seal hides the home folder, and "
            "with it the other entries' folders."
        )
    here = folder / kind
    done = here / "marl-bgs-entry-check" / f"{commit}.json"
    if done.exists() or (kind == "official" and (here / "run").exists()):
        raise SystemExit(
            f"{here} already holds this commit's {kind}; move it aside first."
        )
    linger = ["loginctl", "show-user", str(os.getuid()), "-p", "Linger", "--value"]
    if (
        subprocess.run(
            linger, capture_output=True, text=True, check=False
        ).stdout.strip()
        != "yes"
    ):
        raise SystemExit(
            "Run `loginctl enable-linger` once, so a logout cannot stop the unit."
        )
    try:
        sandbox(folder, online=True)
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    if Path(repository).exists():
        repository = str(Path(repository).resolve())
    short = commit[:12]
    here.mkdir(parents=True, exist_ok=True)
    unit = f"marl-bgs-{kind}-{short}"
    log = here / f"check-{short}.log"
    arguments = [kind, _IN_UNIT, repository, commit, str(folder)]
    status = subprocess.run(
        _unit_command(here, arguments, unit=unit, detach=True, log=log), check=False
    ).returncode
    if status == 0:
        print(f"Started {unit}. Status: systemctl --user status {unit}. Log: {log}")
    return status


def _check_here(kind: str, repository: str, commit: str, folder: Path) -> int:
    """Run the check in this process, sealed; the unit that ``_start`` made calls this.

    The intake test keeps ``check_entry``'s own 15 minutes; the official run
    trains for ``entry.OFFICIAL_TRAIN_MINUTES`` and keeps its folder. The log's
    last line is always the unit's peak RAM (see ``_peak_ram``), even when
    ``check_entry`` raises (a bad SHA, an existing run folder, a stop), which
    then ends the unit with a traceback and no record. Returns 0 when no
    must-pass step failed, else 1.
    """
    here = folder / kind
    try:
        if kind == "official":
            result = entry.check_entry(
                here,
                repository=repository,
                commit=commit,
                train_minutes=entry.OFFICIAL_TRAIN_MINUTES,
                sandbox=sandbox,
                keep_folder=here / "run",
            )
        else:
            result = entry.check_entry(
                here, repository=repository, commit=commit, sandbox=sandbox
            )
    finally:
        print(_peak_ram(), flush=True)
    return 0 if result.passed else 1


def _peak_ram() -> str:
    """Return the peak RAM of this process's cgroup so far, as one line of text.

    Inside the unit, the cgroup is the unit's own, so the peak covers the check
    and every program it started, file cache included (``memory.peak``, cgroup
    v2). Returns "Peak RAM: not measured ..." when the file cannot be read, for
    example outside cgroup v2; it never raises.
    """
    try:
        cgroup = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        peak = Path("/sys/fs/cgroup", cgroup.lstrip("/"), "memory.peak").read_text()
        return f"Peak RAM of this unit: {int(peak) / 1e9:.2f} GB (memory.peak)."
    except (OSError, IndexError, ValueError):
        return "Peak RAM: not measured (no cgroup memory.peak here)."


def _run_here(clone: Path, program: Sequence[str]) -> int:
    """Run ``program`` in ``clone``, sealed and offline; ``run``'s unit calls this.

    The program gets ``entry.entry_environment(clone, offline=True)``, the
    environment ``check_entry`` gives its own offline children (the clone's
    ``.venv`` first on ``PATH`` and its model cache as ``HF_HOME``), behind
    ``sandbox(clone, online=False)``. Returns the program's exit status.
    ``sandbox``'s errors propagate.
    """
    clone = clone.resolve()
    return subprocess.run(
        [*sandbox(clone, online=False), *program],
        cwd=clone,
        env=entry.entry_environment(clone, offline=True),
        check=False,
    ).returncode


def _run(clone: Path, program: Sequence[str]) -> int:
    """Run ``program`` sealed in ``clone`` through a waiting unit; ``run`` calls this.

    The unit is ``marl-bgs-run-<this process's ID>``. Its output comes back to
    this terminal and its input is closed, so the entry's code cannot read what
    the maintainer types. Ctrl-C stops the unit too, instead of leaving it
    running. Returns the program's exit status, or 130 after Ctrl-C.
    """
    unit = f"marl-bgs-run-{os.getpid()}"
    arguments = ["run", _IN_UNIT, str(clone), "--", *program]
    command = _unit_command(clone, arguments, unit=unit, detach=False, log=None)
    try:
        return subprocess.run(command, stdin=subprocess.DEVNULL, check=False).returncode
    except KeyboardInterrupt:
        subprocess.run(["systemctl", "--user", "stop", unit], check=False)
        return 130


def main(argv: Sequence[str] | None = None) -> int:
    """Read the command line and run ``intake``, ``official`` or ``run``.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        The arguments after the script's name; None reads ``sys.argv``.

    Returns
    -------
    int
        0 when ``intake`` or ``official`` started its unit; inside the unit, 0
        when the check passed and 1 when it failed; for ``run``, the command's
        exit status. ``run`` waits for its unit and passes the command's input
        and output through.
    """
    parser = argparse.ArgumentParser(
        description="Run a leaderboard entry's intake test or official run, sealed."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    hours = f"{entry.OFFICIAL_TRAIN_MINUTES / 60:g}"
    for kind, text in (
        ("intake", "start the 15-minute intake test as a systemd user unit"),
        ("official", f"start the {hours}-hour official run as a systemd user unit"),
    ):
        check = commands.add_parser(kind, help=text)
        check.add_argument(_IN_UNIT, action="store_true", help=argparse.SUPPRESS)
        check.add_argument("repository", help="the entry's repository URL or path")
        check.add_argument("commit", help="the full 40-character commit SHA")
        check.add_argument(
            "folder",
            type=Path,
            help="the entry's folder; everything goes into FOLDER/intake or "
            "FOLDER/official",
        )
    run = commands.add_parser(
        "run", help="run one command sealed and offline in a clone"
    )
    run.add_argument(_IN_UNIT, action="store_true", help=argparse.SUPPRESS)
    run.add_argument("clone", type=Path, help="the entry's clone")
    run.add_argument("program", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args(argv)
    if args.command == "run":
        program: list[str] = args.program
        if not program:
            parser.error(
                "give the command after --, for example: run CLONE -- python export.py"
            )
        if args.in_unit:
            return _run_here(args.clone, program)
        return _run(args.clone.resolve(), program)
    if args.in_unit:
        return _check_here(args.command, args.repository, args.commit, args.folder)
    return _start(args.command, args.repository, args.commit, args.folder)


if __name__ == "__main__":
    sys.exit(main())
