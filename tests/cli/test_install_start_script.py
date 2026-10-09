"""Check how the install routes pick the JAX build.

``start.sh``, checked through ``sh start.sh --dry-run`` (it prints the choice
and the two commands, installs nothing and does not install uv):

- ``MARL_BGS_ACCELERATOR`` wins when set (``cpu``, ``cuda13``, ``cuda12``,
  ``rocm``, ``tpu``); any other value stops with status 1 and says which
  values are allowed.
- Without it, ``nvidia-smi`` decides from the lowest driver version and the
  lowest compute capability across all GPUs: driver 580+ and compute 7.5+
  gives ``cuda13``, driver 525+ and compute 5.2+ gives ``cuda12``, anything
  else gives ``cpu``. No ``nvidia-smi``, or one that fails or prints nothing
  usable, gives ``cpu``. ``rocm`` and ``tpu`` are never picked by themselves.
- The sync command is ``uv sync --locked --extra all --group jupyter`` plus
  ``--extra BUILD`` for a GPU build. ``--dry-run`` may stand anywhere on the
  line, and every other option goes to ``play.py`` unchanged, in order.

The notebooks (``examples/quickstart.ipynb`` and
``examples/evaluation_analysis.ipynb``) are saved unexecuted, every code cell
compiles as plain Python, and both start with the same install cell. That cell
installs nothing outside Colab and Kaggle. On them it pip-installs
``marl-battlegrounds[all]==1.0.0`` plus the build of the JAX plugin already
there: ``cuda13``, ``cuda12`` or ``tpu`` (``libtpu``), else none.

One list of names: ``start.sh``'s builds are exactly the GPU extras in
``pyproject.toml`` (plus ``cpu``), and ``play.py``, the Dockerfile and the
quick start notebook name the same six released Season 0 models.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPOSITORY = Path(__file__).resolve().parents[2]
_START = _REPOSITORY / "start.sh"
_NOTEBOOKS = ("quickstart.ipynb", "evaluation_analysis.ipynb")


def _dry_run(
    tmp_path: Path,
    *,
    gpus: str | None = None,
    accelerator: str | None = None,
    smi_status: int = 0,
    options: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("awk", "dirname"):
        found = shutil.which(name)
        assert found is not None
        (tools / name).symlink_to(found)
    if gpus is not None:
        smi = tools / "nvidia-smi"
        smi.write_text(
            f"#!/bin/sh\nprintf '%s' '{gpus}'\nexit {smi_status}\n", encoding="utf-8"
        )
        smi.chmod(0o755)
    environment = {"PATH": str(tools), "HOME": str(tmp_path / "home")}
    if accelerator is not None:
        environment["MARL_BGS_ACCELERATOR"] = accelerator
    shell = shutil.which("sh")
    assert shell is not None
    return subprocess.run(
        [shell, str(_START), *options[:1], "--dry-run", *options[1:]],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("gpus", "build"),
    [
        (None, "cpu"),
        ("580.65.06, 12.0\n", "cuda13"),
        ("580.65.06, 7.5\n", "cuda13"),
        ("575.57.08, 8.9\n", "cuda12"),
        ("525.60.13, 5.2\n", "cuda12"),
        ("580.65.06, 12.0\n580.65.06, 7.0\n", "cuda12"),
        ("520.61.05, 8.6\n", "cpu"),
        ("580.65.06, 5.0\n", "cpu"),
        ("", "cpu"),
    ],
)
def test_nvidia_smi_picks_the_build(
    tmp_path: Path, gpus: str | None, build: str
) -> None:
    if gpus is None and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        pytest.skip("start.sh finds WSL2's own nvidia-smi on this machine")
    result = _dry_run(tmp_path, gpus=gpus)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith(f"MARL-BattleGrounds: using the {build} build (")
    extra = "" if build == "cpu" else f" --extra {build}"
    assert lines[1] == f"uv sync --locked --extra all --group jupyter{extra}"
    assert not (tmp_path / "home").exists()


def test_a_failing_nvidia_smi_means_cpu(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, gpus="", smi_status=9)
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout.splitlines()[1] == "uv sync --locked --extra all --group jupyter"
    )


@pytest.mark.parametrize("build", ["cpu", "cuda13", "cuda12", "rocm", "tpu"])
def test_the_accelerator_setting_wins(tmp_path: Path, build: str) -> None:
    result = _dry_run(tmp_path, gpus="580.65.06, 12.0\n", accelerator=build)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == (
        f"MARL-BattleGrounds: using the {build} build (set by MARL_BGS_ACCELERATOR)."
    )
    extra = "" if build == "cpu" else f" --extra {build}"
    assert lines[1] == f"uv sync --locked --extra all --group jupyter{extra}"


def test_an_unknown_accelerator_stops(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, accelerator="metal")
    assert result.returncode == 1
    assert "cpu, cuda13, cuda12, rocm or tpu" in result.stderr
    assert result.stdout == ""


def test_other_options_go_to_play(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, options=("--team-b", "qmix", "--map", "12"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[2] == (
        "uv run --no-sync python play.py --team-b qmix --map 12"
    )


def _install_cells() -> list[str]:
    cells: list[str] = []
    for name in _NOTEBOOKS:
        path = _REPOSITORY / "examples" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for cell in code:
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), str(path), "exec")
        cells.append("".join(code[0]["source"]))
    return cells


def _run_install_cell(
    monkeypatch: pytest.MonkeyPatch, *, service: str | None, plugins: tuple[str, ...]
) -> list[list[str]]:
    cell = _install_cells()[0]
    calls: list[list[str]] = []

    def record(command: list[str], **_kwargs: object) -> None:
        calls.append(command)

    distributions = [SimpleNamespace(metadata={"Name": name}) for name in plugins]
    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: distributions)
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    if service == "colab":
        monkeypatch.setitem(sys.modules, "google.colab", ModuleType("google.colab"))
    elif service == "kaggle":
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    exec(cell, {})
    return calls


def test_both_notebooks_start_with_the_same_install_cell() -> None:
    first, second = _install_cells()
    assert first == second


@pytest.mark.parametrize(
    ("service", "plugins", "package"),
    [
        (
            "colab",
            ("jax", "jax-cuda13-plugin"),
            "marl-battlegrounds[all,cuda13]==1.0.0",
        ),
        (
            "colab",
            ("jax", "jax-cuda12-plugin"),
            "marl-battlegrounds[all,cuda12]==1.0.0",
        ),
        ("kaggle", ("jax", "libtpu"), "marl-battlegrounds[all,tpu]==1.0.0"),
        ("kaggle", ("jax",), "marl-battlegrounds[all]==1.0.0"),
        (None, ("jax", "jax-cuda13-plugin"), None),
    ],
)
def test_the_install_cell_matches_the_notebook_service(
    monkeypatch: pytest.MonkeyPatch,
    service: str | None,
    plugins: tuple[str, ...],
    package: str | None,
) -> None:
    calls = _run_install_cell(monkeypatch, service=service, plugins=plugins)
    expected = (
        []
        if package is None
        else [[sys.executable, "-m", "pip", "install", "-q", package]]
    )
    assert calls == expected


def test_one_list_of_builds_and_released_models() -> None:
    pyproject = tomllib.loads((_REPOSITORY / "pyproject.toml").read_text())
    conflicts = pyproject["tool"]["uv"]["conflicts"][0]
    start = _START.read_text(encoding="utf-8")
    builds = re.search(r"^ +(cpu \| [a-z0-9 |]+)\)$", start, re.MULTILINE)
    assert builds is not None
    assert set(builds.group(1).split(" | ")) == {"cpu"} | {
        item["extra"] for item in conflicts
    }
    sys.path.insert(0, str(_REPOSITORY))
    try:
        import play
    finally:
        sys.path.remove(str(_REPOSITORY))
    released = {play.RELEASED_PREFIX + name for name in play.TRAINED_TEAMS.values()}
    pattern = r"MARL-BattleGrounds/tdm-season-0-[a-z-]+"
    dockerfile = (_REPOSITORY / "docker" / "Dockerfile").read_text(encoding="utf-8")
    notebook = (_REPOSITORY / "examples" / "quickstart.ipynb").read_text(
        encoding="utf-8"
    )
    assert len(released) == 6
    assert set(re.findall(pattern, dockerfile)) == released
    assert set(re.findall(pattern, notebook)) == released
