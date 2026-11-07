"""Check the seal that scripts/dev/run_entry.py puts around an entry's own code.

``sandbox(clone, online=False)`` runs a small program under bubblewrap on this
machine (skipped where bubblewrap cannot run). Inside the seal the internet,
this computer's loopback ports and its abstract sockets are out of reach; the
real home folder, /run and the outer /tmp are hidden; HOME and TMPDIR point
inside the clone and WANDB_MODE is offline; the program starts in the clone
and can write there. A stop signal sent to the child's process group reaches
the program, not bubblewrap, so the program can still save before it ends.
The install's seal refuses to run until it has a network of its own, and the
script's own environment keeps only PATH, HOME, LANG and CUDA_VISIBLE_DEVICES.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import uuid
from pathlib import Path

import pytest
from scripts.dev import run_entry

PYTHON = "/usr/bin/python3"
INNER = r"""
import json, os, pathlib, signal, socket, sys

home, outside, port, name = sys.argv[1:]


def reaches(family, address):
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(address) == 0


def save(number, frame):
    pathlib.Path("saved").write_text("yes")
    sys.exit(0)


facts = {
    "internet": reaches(socket.AF_INET, ("1.1.1.1", 53)),
    "loopback": reaches(socket.AF_INET, ("127.0.0.1", int(port))),
    "abstract": reaches(socket.AF_UNIX, "\0" + name),
    "home": sorted(os.listdir(home)),
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
    probe = [bwrap, "--ro-bind", "/", "/", "--unshare-net", "true"]
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
    # Only uv may show in the home folder, and only when it lives there.
    assert set(facts.pop("home")) <= {".local"}
    assert facts == {
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


def test_install_seal_refuses_without_its_own_network(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="passt"):
        run_entry.sandbox(tmp_path, online=True)


def test_unit_environment_keeps_only_the_safe_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "PATH": "/usr/bin",
        "HOME": "/home/maintainer",
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "GPU-1",
        "HF_TOKEN": "hf_secret",
        "SSH_AUTH_SOCK": "/run/user/1000/ssh",
        "DISPLAY": ":1",
    }
    monkeypatch.setattr(os, "environ", environment)
    run_entry._clear_environment()  # pyright: ignore[reportPrivateUsage]
    assert environment == {
        "PATH": "/usr/bin",
        "HOME": "/home/maintainer",
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "GPU-1",
    }
