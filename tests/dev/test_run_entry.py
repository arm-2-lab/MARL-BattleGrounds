"""Check the seal that scripts/dev/run_entry.py puts around an entry's own code.

``sandbox(clone, online=False)`` runs a small program under bubblewrap on this
machine (skipped where bubblewrap cannot run). Inside the seal the internet,
this computer's loopback ports, its abstract sockets and its other programs
are out of reach; the real home folder (all but uv's own files), /run and the
outer /tmp are hidden; HOME and TMPDIR point inside the clone and WANDB_MODE
is offline; the program starts in the clone and can write there. The program
starts with Python's default Ctrl-C handler, and a stop signal sent to the
child's process group reaches it, not bubblewrap, so it can still save before
it ends. The seal refuses an install until it has a network of its own, and a
clone outside the home folder and /tmp, where it cannot hide the other
entries. A unit starts with only PATH, HOME, LANG and CUDA_VISIBLE_DEVICES,
whatever the maintainer's shell holds, and systemd does not rewrite ``$`` in
its arguments.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from scripts.dev import run_entry

# The repository's own Python lives under the home folder, which the seal hides.
PYTHON = "/usr/bin/python3"
INNER = r"""
import json, os, pathlib, signal, socket, sys

home, outside, port, name, host_pid = sys.argv[1:]


def reaches(family, address):
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(address) == 0


def signals(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def listing(*parts):
    path = os.path.join(home, *parts)
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


def save(number, frame):
    pathlib.Path("saved").write_text("yes")
    sys.exit(0)


facts = {
    "internet": reaches(socket.AF_INET, ("1.1.1.1", 53)),
    "loopback": reaches(socket.AF_INET, ("127.0.0.1", int(port))),
    "abstract": reaches(socket.AF_UNIX, "\0" + name),
    "home": listing(),
    "home_local": listing(".local"),
    "home_share": listing(".local", "share"),
    "host_process": signals(host_pid),
    "default_ctrl_c": signal.getsignal(signal.SIGINT) is signal.default_int_handler,
    "outside": os.path.exists(outside),
    "run": os.listdir("/run"),
    "env": {key: os.environ.get(key) for key in ("HOME", "TMPDIR", "WANDB_MODE")},
    "cwd": os.getcwd(),
}
pathlib.Path("facts.json").write_text(json.dumps(facts))
signal.signal(signal.SIGINT, save)
print("ready", flush=True)
signal.pause()
"""


def _seal_runs() -> bool:
    bwrap = shutil.which("bwrap")
    if bwrap is None or not Path(PYTHON).exists():
        return False
    probe = [bwrap, "--ro-bind", "/", "/", "--unshare-all", "true"]
    return subprocess.run(probe, capture_output=True, check=False).returncode == 0


@pytest.mark.skipif(not _seal_runs(), reason="bubblewrap cannot run here")
def test_offline_seal_hides_this_computer_and_passes_the_stop_signal(
    tmp_path: Path,
) -> None:
    clone = tmp_path / "entry"
    clone.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    name = f"marl-bgs-seal-test-{uuid.uuid4().hex}"
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix,
    ):
        tcp.bind(("127.0.0.1", 0))
        tcp.listen()
        unix.bind("\0" + name)
        unix.listen()
        port = str(tcp.getsockname()[1])
        child = subprocess.Popen(
            [
                *run_entry.sandbox(clone, online=False),
                *(PYTHON, "-c", INNER, str(Path.home()), str(outside), port, name),
                str(os.getpid()),
            ],
            cwd=clone,
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert child.stdout is not None
        assert child.stdout.readline() == "ready\n"
        os.killpg(child.pid, signal.SIGINT)
        assert child.wait() == 0
    assert (clone / "saved").read_text() == "yes"
    facts = json.loads((clone / "facts.json").read_text())
    # Only uv and its Python installs may show in the home folder, read-only.
    assert set(facts.pop("home")) <= {".local"}
    assert set(facts.pop("home_local")) <= {"bin", "share"}
    assert set(facts.pop("home_share")) <= {"uv"}
    assert facts == {
        "host_process": False,
        "default_ctrl_c": True,
        "internet": False,
        "loopback": False,
        "abstract": False,
        "outside": False,
        "run": [],
        "env": {
            "HOME": str(clone / ".entry-check-home"),
            "TMPDIR": "/tmp",
            "WANDB_MODE": "offline",
        },
        "cwd": str(clone),
    }


def test_seal_refuses_an_install_and_a_clone_it_cannot_hide(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="passt"):
        run_entry.sandbox(tmp_path, online=True)
    with pytest.raises(RuntimeError, match="other entries"):
        run_entry.sandbox(Path("/srv/entry"), online=False)


def test_unit_starts_with_only_the_four_safe_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1")
    command = run_entry._unit_command(  # pyright: ignore[reportPrivateUsage]
        Path("/tmp/entry"), ["intake"], unit="unit", detach=True, log=None
    )
    variables = command[command.index("-i") + 1 : command.index(sys.executable)]
    names = [variable.split("=", 1)[0] for variable in variables]
    assert names == ["PATH", "HOME", "LANG", "CUDA_VISIBLE_DEVICES"]
    assert "CUDA_VISIBLE_DEVICES=GPU-1" in variables
    assert not any("hf_secret" in part for part in command)
    assert "--expand-environment=no" in command
