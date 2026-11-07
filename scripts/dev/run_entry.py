"""Run leaderboard entries sealed off from the network and this computer's secrets.

Maintainers use this script for the intake test and the official run of each
leaderboard entry (the private tournament protocol says when). Both are one
call of ``marl_battlegrounds.entry.check_entry``, the same check entrants run
at home; this script adds the seal and the limits around it. Run it from a
checkout whose ``check_entry`` takes ``sandbox`` and ``keep_folder``, and name
the card in ``CUDA_VISIBLE_DEVICES``::

    CUDA_VISIBLE_DEVICES=GPU-<uuid> uv run --no-sync python scripts/dev/run_entry.py \\
        intake REPOSITORY COMMIT FOLDER
    CUDA_VISIBLE_DEVICES=GPU-<uuid> uv run --no-sync python scripts/dev/run_entry.py \\
        official REPOSITORY COMMIT FOLDER
    CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/dev/run_entry.py \\
        run CLONE -- COMMAND [ARGUMENT ...]

- ``intake`` checks COMMIT of REPOSITORY with 15 minutes of training, in a
  temporary folder that the check removes.
- ``official`` is the same check with 8 hours of training, in
  ``FOLDER/official/run``, which must not exist yet and is kept: the clone is
  ``FOLDER/official/run/entry``, with the run's output and its model cache.
- Both start a systemd user unit, ``marl-bgs-intake-<commit>`` or
  ``marl-bgs-official-<commit>`` (the commit's first 12 characters), and return
  at once. Each kind has its own folder, ``FOLDER/intake`` or
  ``FOLDER/official``, so the two records of one commit never overwrite each
  other: the check's output goes to ``check-<commit>.log`` there, and its
  record to ``marl-bgs-entry-check/<commit>.json`` there. ``systemctl --user status
  <unit>`` shows it; ``systemctl --user stop <unit>`` sends it SIGINT; after it
  ends, ``journalctl --user -u <unit>`` keeps systemd's account of how.
- ``run`` runs one command in an entry's clone, sealed and offline, and waits
  for it. The protocol uses it to export the last saved checkpoint when the
  official run saved no choice. Only the clone is writable inside the seal, so
  the command must write its file inside the clone (for example
  ``CLONE/entry.zip``); a file written anywhere else is lost.

The seal is ``sandbox``'s prefix, put in front of every program that runs the
entry's own code. The limits come from the unit: 48 GB of RAM and no swap for
everything inside it, SIGINT then 120 seconds before a kill on a stop, and a
stop after 11 hours. A program past 48 GB is killed by the kernel and the rest
of the unit carries on (``OOMPolicy=continue``), so the check still records
what happened. The 11 hours are a backstop above the check's own timers (1
hour to install and download, the training time, 10 minutes each to load and
export, 30 minutes of play), so systemd never stops a healthy run. Inside the
unit the environment keeps only ``PATH``, ``HOME``, ``LANG`` and
``CUDA_VISIBLE_DEVICES``, so no token or key in the maintainer's shell reaches
the check or the entry. When the unit ends, ``journalctl --user -u <unit>``
keeps its "Consumed ... memory peak" line, the unit's peak RAM with file cache
included. systemd-oomd may still kill a unit whose memory is under heavy
pressure before 48 GB (this machine's user manager kills at 50% pressure);
``journalctl -u systemd-oomd`` then names it.

Prerequisites: Linux with bubblewrap (``bwrap``), pasta (``passt``), systemd
user units and uv on ``PATH``. Run ``loginctl enable-linger`` once, so that a
unit keeps running after a logout.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from marl_battlegrounds import entry

_LIMITS = (
    "--property=MemoryMax=48G",
    "--property=MemorySwapMax=0",
    "--property=RuntimeMaxSec=11h",
    "--property=KillSignal=SIGINT",
    "--property=TimeoutStopSec=120",
    "--property=OOMPolicy=continue",
)
_KEPT_VARIABLES = ("PATH", "HOME", "LANG", "CUDA_VISIBLE_DEVICES")
_IN_UNIT = "--in-unit"
_NO_INSTALL_SEAL = (
    "The install's seal needs a network of its own, which is not set up on this "
    "machine yet: install pasta (sudo apt install passt) and prove it first. "
    "Nothing was run."
)


def sandbox(clone: Path, online: bool) -> list[str]:
    """Return the command prefix that seals one of the entry's own programs.

    ``check_entry`` puts it in front of every child that runs the entry's code,
    and ``run`` uses it too. Inside, the program sees the whole file system
    read-only, the NVIDIA devices, and ``clone`` writable. This computer's home
    folder, ``/run`` (Docker's socket, D-Bus, the SSH agent and the desktop's
    sockets) and the real ``/tmp`` are hidden, and the network is cut, which
    also cuts this computer's loopback ports and abstract sockets, such as the
    desktop's display. If uv and its Python installs (``uv python dir``) live
    in the home folder, they alone are shown there, read-only: an environment
    that uv built outside the seal links its ``python`` to one of them.

    Parameters
    ----------
    clone : Path
        The entry's clone. Two folders are made inside it if missing:
        ``.entry-check-home``, which becomes ``HOME``, and ``.entry-check-tmp``,
        which becomes ``/tmp``. So caches and temporary files count as the
        entry's disk, not RAM.
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
        That needs pasta (``sudo apt install passt``) and a proof on this
        machine, which are not done yet, so nothing is run. Also raised when
        bubblewrap is missing.

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
    home, tmp = clone / ".entry-check-home", clone / ".entry-check-tmp"
    home.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)
    real_home = Path.home()
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
        *(
            arg
            for device in sorted(glob.glob("/dev/nvidia*"))
            for arg in ("--dev-bind", device, device)
        ),
        *("--proc", "/proc", "--tmpfs", "/run", "--tmpfs", str(real_home)),
        *(arg for path in shown for arg in ("--ro-bind", path, path)),
        # /tmp comes before the clone, because the clone may live under /tmp.
        *("--bind", str(tmp), "/tmp", "--bind", str(clone), str(clone)),
        "--unshare-net",
        *("--setenv", "HOME", str(home), "--setenv", "TMPDIR", "/tmp"),
        *("--setenv", "WANDB_MODE", "offline"),
        *("env", "--default-signal=INT"),
    ]


def _clear_environment() -> None:
    """Keep only PATH, HOME, LANG and CUDA_VISIBLE_DEVICES in this environment.

    Runs inside the unit before ``check_entry``, whose children inherit the
    environment (a user unit starts with the session manager's variables, not
    the maintainer's shell's).
    """
    kept = {name: os.environ[name] for name in _KEPT_VARIABLES if name in os.environ}
    os.environ.clear()
    os.environ.update(kept)


def _tool_path() -> str:
    """Return a PATH for the unit: uv's folder, then the usual system folders."""
    uv = shutil.which("uv")
    folders = ([str(Path(uv).parent)] if uv else []) + [
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ]
    return os.pathsep.join(dict.fromkeys(folders))


def _start(kind: str, repository: str, commit: str, folder: Path) -> int:
    """Start the intake test or the official run as a systemd user unit.

    Parameters
    ----------
    kind : {"intake", "official"}
        Which check: 15 minutes of training in a temporary folder, or 8 hours
        kept in ``folder / "official" / "run"``.
    repository, commit : str
        What to check: a repository URL or path, and the full commit SHA.
        ``check_entry`` checks the SHA's form inside the unit.
    folder : Path
        The entry's folder. ``folder / kind`` is made if missing and gets the
        log ``check-<commit>.log`` (the commit's first 12 characters, emptied at
        each start), the record and, for the official run, the run itself.

    Returns
    -------
    int
        0 once the unit has started; the check itself is still running.

    Raises
    ------
    SystemExit
        ``CUDA_VISIBLE_DEVICES`` is not set, so the card would not be chosen; or
        pasta is missing, so every check would fail at step 4's install for a
        reason of ours (see ``sandbox``). No unit starts.
    subprocess.CalledProcessError
        systemd refused the unit, for example because a unit of the same name
        is still running.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES to the card the protocol names.")
    if shutil.which("pasta") is None:
        raise SystemExit(f"No unit started. {_NO_INSTALL_SEAL}")
    folder = folder.resolve()
    (folder / kind).mkdir(parents=True, exist_ok=True)
    unit = f"marl-bgs-{kind}-{commit[:12]}"
    log = folder / kind / f"check-{commit[:12]}.log"
    subprocess.run(
        [
            *("systemd-run", "--user", f"--unit={unit}", "--collect", "--quiet"),
            *_LIMITS,
            f"--property=StandardOutput=truncate:{log}",
            f"--working-directory={folder / kind}",
            f"--setenv=PATH={_tool_path()}",
            "--setenv=LANG=C.UTF-8",
            f"--setenv=CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
            *(sys.executable, "-u", str(Path(__file__).resolve())),
            *(kind, repository, commit, str(folder), _IN_UNIT),
        ],
        check=True,
    )
    print(f"Started {unit}. Status: systemctl --user status {unit}. Log: {log}")
    return 0


def _check_here(kind: str, repository: str, commit: str, folder: Path) -> int:
    """Run the check in this process, sealed; the unit that ``_start`` made calls this.

    The intake test keeps ``check_entry``'s own 15 minutes; the official run
    trains for ``entry.OFFICIAL_TRAIN_MINUTES`` and keeps its folder. Returns 0
    when no must-pass step failed, else 1. The record and the kept folder are
    as described at the top of this file.
    """
    _clear_environment()
    here = folder / kind
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
    return 0 if result.passed else 1


def _run_sealed(clone: Path, command: Sequence[str]) -> int:
    """Run one command in an entry's clone, sealed and offline, and wait for it.

    The command runs in a systemd user unit with the same limits as a check,
    and with only these variables: ``LANG`` and ``CUDA_VISIBLE_DEVICES`` from
    the caller (empty hides the GPU), ``PATH`` with the clone's ``.venv/bin``
    first, ``VIRTUAL_ENV``, ``HF_HOME`` set to the clone's
    ``.entry-check-hf`` (the model cache the check filled), and
    ``HF_HUB_OFFLINE``, ``UV_OFFLINE`` and ``UV_NO_SYNC`` set to 1. Only the
    clone is writable, so the command must write its output inside it.

    Returns
    -------
    int
        The command's exit status, or 2 when no command was given.
    """
    if not command:
        print("Give the command after --, for example: run CLONE -- python export.py")
        return 2
    clone = clone.resolve()
    venv = clone / ".venv"
    variables = {
        name: os.environ[name]
        for name in ("LANG", "CUDA_VISIBLE_DEVICES")
        if name in os.environ
    }
    variables.update(
        PATH=f"{venv / 'bin'}{os.pathsep}{_tool_path()}",
        HOME=str(Path.home()),
        VIRTUAL_ENV=str(venv),
        HF_HOME=str(clone / ".entry-check-hf"),
        HF_HUB_OFFLINE="1",
        UV_OFFLINE="1",
        UV_NO_SYNC="1",
    )
    finished = subprocess.run(
        [
            *("systemd-run", "--user", "--wait", "--pipe", "--collect", "--quiet"),
            *_LIMITS,
            f"--working-directory={clone}",
            *("env", "-i", *(f"{name}={value}" for name, value in variables.items())),
            *sandbox(clone, online=False),
            *command,
        ],
        check=False,
    )
    return finished.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Read the command line and run ``intake``, ``official`` or ``run``.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        The arguments after the script's name; None reads ``sys.argv``.

    Returns
    -------
    int
        The exit status: 0 when a unit started or, inside the unit, when the
        check passed; otherwise as each command says.
    """
    parser = argparse.ArgumentParser(
        description="Run a leaderboard entry's intake test or official run, sealed."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for kind, text in (
        ("intake", "start the 15-minute intake test as a systemd user unit"),
        (
            "official",
            "start the 8-hour official run as a systemd user unit, kept in "
            "FOLDER/official/run",
        ),
    ):
        check = commands.add_parser(kind, help=text)
        check.add_argument("repository", help="the entry's repository URL or path")
        check.add_argument("commit", help="the full 40-character commit SHA")
        check.add_argument(
            "folder", type=Path, help="the entry's folder; FOLDER/<command> gets it all"
        )
        check.add_argument(_IN_UNIT, action="store_true", help=argparse.SUPPRESS)
    run = commands.add_parser(
        "run", help="run one command sealed and offline in a clone"
    )
    run.add_argument("clone", type=Path, help="the entry's clone")
    run.add_argument("program", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args(argv)
    if args.command == "run":
        program: list[str] = args.program
        return _run_sealed(
            args.clone, program[1:] if program[:1] == ["--"] else program
        )
    if args.in_unit:
        return _check_here(args.command, args.repository, args.commit, args.folder)
    return _start(args.command, args.repository, args.commit, args.folder)


if __name__ == "__main__":
    sys.exit(main())
