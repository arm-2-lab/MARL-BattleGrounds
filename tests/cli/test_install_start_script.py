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

``start.sh`` without uv, run for real with stand-ins for ``curl``, uv's
installer and uv: the installer runs while ``~/.local/bin`` is not yet on
``PATH`` (uv's installer writes that folder into the shell startup files, so
new terminals find ``uv``, only when the folder is not on ``PATH``). A uv
already in ``~/.local/bin`` but not on ``PATH`` is found and used, with no
installer run. An installer that exits with an error after putting uv in place
(as uv's does when ``~/.config`` is not writable) does not stop ``start.sh``: it
says uv is installed and goes on to the sync. When uv is still missing, it
stops with status 1, names the internet and home-folder causes and gives the
``sudo chown -R "$USER" ~/.config ~/.local`` fix.

Every tracked path is at most 150 characters, so ``git clone`` works with Git
for Windows: it keeps ``core.longpaths`` off, so a full path, including the
folder the clone goes in, stops at 260 characters.

The notebooks (``examples/quickstart.ipynb`` and
``examples/evaluate/evaluation_analysis.ipynb``) are saved with the outputs of one
complete run, so readers see every result before running anything: every code
cell ran once, in order, with no error output. Every code cell compiles as plain
Python, and both start with the same install cell. That cell
installs nothing outside Colab. In Colab, when its ``VERSION`` (always the
package's own version) is already installed, it installs nothing; otherwise it
pip-installs ``marl-battlegrounds[all]==VERSION`` plus the build of the JAX
plugin already there: ``cuda13``, ``cuda12`` or ``tpu`` (``libtpu``), else
none. It then carries on without a restart, unless pip replaced a package
that is already loaded or JAX fails to import and run; only then it restarts
the runtime with ``exit(0)``. It restarts only when a loaded module's file is
one pip just wrote: a package Colab keeps twice that pip left alone, or a new
file in a shared folder loaded from elsewhere, or a replaced debugger plugin
(``pydevd_plugins``, such as omegaconf's), does not count.

One list of names: ``start.sh``'s builds are exactly the GPU extras in
``pyproject.toml`` (plus ``cpu``), ``play.py`` and the Dockerfile name the same
six released Season 0 models, the Dockerfile also downloads the released
training-opponent pool, and every released model the quick start notebook
names is one of the six.

``play.py`` prints a plain message instead of a traceback when the
BattleClient refuses a setting (status 2) or the system refuses (status 1),
for example after a mistyped or released team name. It keeps the traceback
when a team runs your own code or files: a ``module:function`` factory or a
file or folder that exists.
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

from marl_battlegrounds.released_models import RESULTS_DATASET, TRAINING_OPPONENTS

_REPOSITORY = Path(__file__).resolve().parents[2]
_START = _REPOSITORY / "start.sh"
_NOTEBOOKS = ("quickstart.ipynb", "evaluate/evaluation_analysis.ipynb")


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


def _run_needing_uv(
    tmp_path: Path,
    *,
    earlier_uv: bool,
    installer_exit: int = 0,
    installs_uv: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("awk", "dirname", "sh", "cat", "mkdir", "chmod"):
        found = shutil.which(name)
        assert found is not None
        (tools / name).symlink_to(found)
    record = tmp_path / "path-while-installing"
    # A stand-in uv that fails its sync, so start.sh stops before play.py.
    make_uv = (
        'mkdir -p "$HOME/.local/bin"\n'
        "printf '#!/bin/sh\\nexit 3\\n' > \"$HOME/.local/bin/uv\"\n"
        'chmod +x "$HOME/.local/bin/uv"\n'
    )
    if earlier_uv:
        subprocess.run(["sh", "-c", make_uv], env={"HOME": str(home)}, check=True)
    else:
        # A stand-in for uv's installer, which curl hands to sh.
        installer = (
            f"printf '%s' \"$PATH\" > '{record}'\n"
            + (make_uv if installs_uv else "")
            + f"exit {installer_exit}\n"
        )
        curl = tools / "curl"
        curl.write_text(f"#!/bin/sh\ncat <<'END'\n{installer}END\n", encoding="utf-8")
        curl.chmod(0o755)
    shell = shutil.which("sh")
    assert shell is not None
    result = subprocess.run(
        [shell, str(_START)],
        env={"PATH": str(tools), "HOME": str(home), "MARL_BGS_ACCELERATOR": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    return result, record, local_bin


def test_uv_installer_runs_before_its_folder_is_on_path(tmp_path: Path) -> None:
    result, record, local_bin = _run_needing_uv(tmp_path, earlier_uv=False)
    assert "Installing uv" in result.stdout
    assert str(local_bin) not in record.read_text(encoding="utf-8").split(":")
    assert result.returncode == 1
    assert "The install failed" in result.stderr


def test_start_carries_on_when_the_installer_fails_after_uv(tmp_path: Path) -> None:
    result, _, _ = _run_needing_uv(tmp_path, earlier_uv=False, installer_exit=1)
    assert "uv is installed. Its installer stopped" in result.stdout
    # It went on to the sync, which the stand-in uv fails.
    assert "The install failed" in result.stderr
    assert result.returncode == 1


def test_a_failed_uv_install_names_the_folder_fix(tmp_path: Path) -> None:
    result, _, _ = _run_needing_uv(
        tmp_path, earlier_uv=False, installer_exit=1, installs_uv=False
    )
    assert "Could not install uv" in result.stderr
    assert 'sudo chown -R "$USER" ~/.config ~/.local' in result.stderr
    assert "The install failed" not in result.stderr
    assert result.returncode == 1


def test_every_path_fits_a_git_for_windows_clone() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert tracked
    assert [path for path in tracked if len(path) > 150] == []


def test_an_earlier_uv_is_found_without_the_installer(tmp_path: Path) -> None:
    result, record, _ = _run_needing_uv(tmp_path, earlier_uv=True)
    assert "Installing uv" not in result.stdout
    assert not record.exists()
    assert result.returncode == 1
    assert "The install failed" in result.stderr


def _install_cells() -> list[str]:
    cells: list[str] = []
    for name in _NOTEBOOKS:
        path = _REPOSITORY / "examples" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert [cell["execution_count"] for cell in code] == list(
            range(1, len(code) + 1)
        )
        for cell in code:
            assert all(output["output_type"] != "error" for output in cell["outputs"])
            compile("".join(cell["source"]), str(path), "exec")
        cells.append("".join(code[0]["source"]))
    return cells


def _run_install_cell(
    monkeypatch: pytest.MonkeyPatch,
    *,
    colab: bool,
    plugins: tuple[str, ...] = ("jax",),
    installed: str | None = None,
    replaced: tuple[str, ...] = (),
    shared: tuple[str, ...] = (),
    twice: tuple[str, ...] = (),
    loaded: dict[str, str] | None = None,
    jax_imports: bool = True,
) -> tuple[list[list[str]], bool]:
    cell = _install_cells()[0]
    calls: list[list[str]] = []
    # Start JAX before the fakes below: its first run looks up its plugins.
    importlib.import_module("jax").numpy.zeros(1).block_until_ready()

    def record(command: list[str], **_kwargs: object) -> None:
        calls.append(command)

    def version(name: str) -> str:
        if installed is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed

    def release(
        name: str, number: str, root: str, extra: tuple[str, ...] = ()
    ) -> SimpleNamespace:
        files = [f"{name}/__init__.py", *(f"{folder}/__init__.py" for folder in extra)]

        def locate_file(file: str) -> str:
            return f"{root}/{file}"

        return SimpleNamespace(
            metadata={"Name": name},
            version=number,
            files=files,
            locate_file=locate_file,
        )

    def distributions() -> list[SimpleNamespace]:
        after = "2" if calls else "1"
        rows = [release(n, "1", "/site") for n in plugins]
        # Two installed copies with different versions, both untouched by pip.
        rows += [
            release(n, v, r) for n in twice for v, r in (("1", "/sys"), ("9", "/site"))
        ]
        # Replaced in place; it may also add a file to a shared folder.
        return rows + [release(n, after, "/site", shared) for n in replaced]

    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.setattr(importlib.metadata, "distributions", distributions)
    monkeypatch.setattr(importlib.metadata, "version", version)
    for name, file in (loaded or {}).items():
        module = ModuleType(name)
        module.__file__ = file
        monkeypatch.setitem(sys.modules, name, module)
    if not jax_imports:
        monkeypatch.setitem(sys.modules, "jax", None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    if colab:
        monkeypatch.setitem(sys.modules, "google.colab", ModuleType("google.colab"))
    try:
        exec(cell, {})
    except SystemExit as stop:
        assert stop.code == 0
        return calls, True
    return calls, False


def _package_version() -> str:
    pyproject = tomllib.loads((_REPOSITORY / "pyproject.toml").read_text())
    return pyproject["project"]["version"]


def test_both_notebooks_start_with_the_same_install_cell() -> None:
    first, second = _install_cells()
    assert first == second


@pytest.mark.parametrize(
    ("plugins", "extras"),
    [
        (("jax", "jax-cuda13-plugin"), "all,cuda13"),
        (("jax", "jax-cuda12-plugin"), "all,cuda12"),
        (("jax", "libtpu"), "all,tpu"),
        (("jax",), "all"),
    ],
)
def test_colab_installs_the_matching_build_and_carries_on(
    monkeypatch: pytest.MonkeyPatch, plugins: tuple[str, ...], extras: str
) -> None:
    calls, restarted = _run_install_cell(
        monkeypatch, colab=True, plugins=plugins, replaced=("newdep",)
    )
    package = f"marl-battlegrounds[{extras}]=={_package_version()}"
    assert calls == [[sys.executable, "-m", "pip", "install", "-q", package]]
    assert not restarted


def test_colab_restarts_when_pip_replaced_a_loaded_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, restarted = _run_install_cell(
        monkeypatch,
        colab=True,
        replaced=("loadeddep",),
        loaded={"loadeddep": "/site/loadeddep/__init__.py"},
    )
    assert len(calls) == 1
    assert restarted


def test_colab_ignores_a_loaded_package_kept_twice_that_pip_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, restarted = _run_install_cell(
        monkeypatch,
        colab=True,
        twice=("systemdep",),
        loaded={"systemdep": "/sys/systemdep/__init__.py"},
    )
    assert len(calls) == 1
    assert not restarted


def test_colab_ignores_a_new_file_in_a_shared_folder_loaded_from_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, restarted = _run_install_cell(
        monkeypatch,
        colab=True,
        replaced=("newdep",),
        shared=("pydevd_plugins",),
        loaded={"pydevd_plugins": "/debugger/pydevd_plugins/__init__.py"},
    )
    assert len(calls) == 1
    assert not restarted


def test_colab_ignores_a_replaced_debugger_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = "pydevd_plugins.extensions.pydevd_plugin_newdep"
    calls, restarted = _run_install_cell(
        monkeypatch,
        colab=True,
        replaced=("newdep",),
        shared=("pydevd_plugins/extensions/pydevd_plugin_newdep",),
        loaded={
            plugin: "/site/pydevd_plugins/extensions/pydevd_plugin_newdep/__init__.py"
        },
    )
    assert len(calls) == 1
    assert not restarted


def test_colab_restarts_when_jax_does_not_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, restarted = _run_install_cell(monkeypatch, colab=True, jax_imports=False)
    assert len(calls) == 1
    assert restarted


@pytest.mark.parametrize(
    ("colab", "installed"), [(False, None), (True, "VERSION"), (False, "VERSION")]
)
def test_the_install_cell_does_nothing_when_not_needed(
    monkeypatch: pytest.MonkeyPatch, colab: bool, installed: str | None
) -> None:
    version = _package_version() if installed else None
    calls, restarted = _run_install_cell(monkeypatch, colab=colab, installed=version)
    assert calls == []
    assert not restarted


def _play_module() -> ModuleType:
    sys.path.insert(0, str(_REPOSITORY))
    try:
        import play
    finally:
        sys.path.remove(str(_REPOSITORY))
    return play


def test_one_list_of_builds_and_released_models() -> None:
    pyproject = tomllib.loads((_REPOSITORY / "pyproject.toml").read_text())
    conflicts = pyproject["tool"]["uv"]["conflicts"][0]
    start = _START.read_text(encoding="utf-8")
    builds = re.search(r"^ +(cpu \| [a-z0-9 |]+)\)$", start, re.MULTILINE)
    assert builds is not None
    assert set(builds.group(1).split(" | ")) == {"cpu"} | {
        item["extra"] for item in conflicts
    }
    play = _play_module()
    released = {play.RELEASED_PREFIX + name for name in play.TRAINED_TEAMS.values()}
    pool = f"{RESULTS_DATASET}/{TRAINING_OPPONENTS}"
    pattern = r"MARL-BattleGrounds/tdm-season-0-[a-z-]+"
    dockerfile = (_REPOSITORY / "docker" / "Dockerfile").read_text(encoding="utf-8")
    notebook = (_REPOSITORY / "examples" / "quickstart.ipynb").read_text(
        encoding="utf-8"
    )
    assert len(released) == 6
    assert set(re.findall(pattern + "(?:/[a-z-]+)?", dockerfile)) == released | {pool}
    named = set(re.findall(pattern, notebook))
    assert named and named <= released


@pytest.mark.parametrize(
    ("team_b", "error", "status"),
    [
        ("qmx", ValueError, 2),
        ("mappo", ValueError, 2),
        ("mappo", OSError, 1),
        ("my_teams:make", ValueError, None),
        ("saved-model", ValueError, None),
    ],
)
def test_play_keeps_the_traceback_of_your_own_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    team_b: str,
    error: type[Exception],
    status: int | None,
) -> None:
    import marl_battlegrounds

    def refuse(**_: object) -> int:
        raise error("refused")

    monkeypatch.setattr(marl_battlegrounds, "battle_client", refuse)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "saved-model").mkdir()
    play = _play_module()
    arguments = ["--team-b", team_b, "--no-browser"]
    if status is None:
        with pytest.raises(error, match="refused"):
            play.main(arguments)
    else:
        assert play.main(arguments) == status
        assert "play.py: refused" in capsys.readouterr().err
