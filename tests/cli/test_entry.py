"""Check the leaderboard entry commands in marl_battlegrounds.entry.

new_entry writes a starter whose files parse: pyproject.toml pins the installed
marl-battlegrounds version with the CUDA 13 build on Linux only,
.python-version is 3.14, .gitignore keeps runs/ out, my_entry defines load,
and entry.yaml passes the same validation entry check uses. Its train line
composes through the packaged Hydra configs into the released Recurrent MAPPO
recipe (250,675,200 steps, 512 games, every actor kept), and its README states
that budget. new_entry prints the method's Hugging Face repo name. Without uv
it prints the uv install line and the note that an installer ERROR about
~/.config still leaves uv installed, and writes no uv.lock.

entry.yaml's name is the method name, and its repo part is the name in
lowercase with every other run of characters made one hyphen ("My_MAPPO" gives
"my-mappo"); an empty or over-64-character part is refused. starting_models
lists downloads as strings, released names or hf:// addresses with a full
commit; 1.0.3's {model, sha256} and {model, commit} items are read by model
alone, paths inside the repository are left out (a released name is kept even
when a folder of that name exists), and 1.0.3's training_maps and config keys
are accepted. Anything else in the list, a bad step suffix, an address without
a full commit, an unknown key, a load that is not module:function and a file
that is not valid YAML are refused with ValueError.

Step 4 accepts a uv.lock only when every marl-battlegrounds in it is a release
from PyPI's index, with each file that version's own file on PyPI's file
server; any released version passes, older or newer (1.0.0, 1.0.1, 1.2.0). It
refuses a missing package, a Git, folder or other-index source, and a file
elsewhere or of another package. A missing or bad uv.lock fails step 4 before
anything is installed. uv sync runs behind sandbox(clone, True). A released
name downloads through the entry's own models download into the check's
HF_HOME, behind sandbox(clone, True) with the caller's offline settings
removed, then once more offline behind sandbox(clone, False). An outside
hf:// address downloads in a child of the check's own isolated Python, behind
no sandbox, and that child's code fails a path with no files. A failed
download fails step 4, and so does running past step 4's hour. Children in
steps 5 to 9 run behind sandbox(clone, False) with exactly
entry_environment(clone, offline=True), HF_HOME and HF_HUB_OFFLINE included,
and step 7's sealed export writes inside the clone before the check copies it
to the work folder's model.zip; a link there is refused, so the check never
copies a file the link points to. A built-in run's choice is read by a sealed
child (its code reports an own loop as no built-in run), never by the check
itself. A committed link that points outside the repository fails step 2,
while one that stays inside passes.

After step 5 the waiting rule holds: a run stopped at its time limit with
nothing saved to enter (an empty own-loop output folder, or a built-in run
without a choice) makes steps 6, 7 and 9 wait and step 8 "not measured"; a run
that ended by itself with nothing saved fails step 6; a saved run makes step 8
pass even when loading fails; an error in step 9 fails step 9 instead of
crashing the check. Step 9 plays the check's copy of the export in a child of
the check's own isolated Python with deterministic XLA, and passes or fails
only a reading on an RTX 5090 that reports its memory; a reading on another
GPU, one without memory, and a CPU are "not measured".

check_entry sorts steps 1-7 as must pass and 8-9 as measured here: a failed
measured step never fails the check, a failed must-pass step does. A dirty
tree fails step 1, skips the rest and saves no record. With install, training
and the later steps stubbed (no uv, no training), a clean commit really
clones, validates entry.yaml and saves its record under .git, where
submit_entry reads it. keep_folder keeps the clone and refuses a folder that
already exists. The machine label says no GPU is visible when
CUDA_VISIBLE_DEVICES hides them all.

submit_entry refuses a repository with no commit, uncommitted changes, the
starter's unchanged train command, a commit on no remote branch and a commit
without a passing check. A user name or token inside the remote's address
never reaches the link. Its link fills exactly the form's text fields (read
from .github/ISSUE_TEMPLATE/leaderboard_submission.yml) plus the title.
play_limits plays 2 self-play games on the CPU through the real evaluate path,
reports them as not a GPU measurement, and refuses an odd number of games.

While entry check runs, SIGTERM and SIGHUP raise KeyboardInterrupt like Ctrl-C,
the running child's whole process group is killed, and the old signal handlers
come back afterwards.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import ast
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import urllib.parse
import zipfile
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
import yaml

from marl_battlegrounds import entry

FORM = Path(__file__).parents[2] / ".github/ISSUE_TEMPLATE/leaderboard_submission.yml"
POOL = "MARL-BattleGrounds/tdm-season-0-results/training-opponents"
QMIX = "MARL-BattleGrounds/tdm-season-0-qmix@final"
OUTSIDE = f"hf://someone/big-model@{'c' * 40}/weights"
LOCK = """\
version = 1
[[package]]
name = "marl-battlegrounds"
version = "1.0.1"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/p/marl_battlegrounds-1.0.1.tar.gz" }
[[package]]
name = "jax-cuda13-plugin"
version = "0.8.0"
"""


def _git(folder: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=folder, capture_output=True, text=True, check=True
    ).stdout.strip()


def _write_entry(folder: Path, **change: object) -> None:
    path = folder / "entry.yaml"
    path.write_text(yaml.safe_dump({**yaml.safe_load(path.read_text()), **change}))


def _checker(folder: Path, sandbox: entry.Sandbox | None = None) -> entry._Checker:
    checker = entry._Checker(str(folder), None, folder.parent, 0.05, None, sandbox)
    checker.clone, checker.entry = folder, entry._read_entry(folder)
    checker.hf_home = folder / ".entry-check-hf"
    return checker


def _form_text_fields() -> set[str]:
    body = yaml.safe_load(FORM.read_text())["body"]
    return {f["id"] for f in body if f["type"] in ("input", "textarea")}


def _own_loop(checker: entry._Checker) -> None:
    checker.builtin_run, checker.choice = False, None


@pytest.fixture
def starter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Path:
    config = tmp_path / "gitconfig"
    config.write_text("[user]\n\tname = Test\n\temail = test@example.com\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    real_which = shutil.which

    def which_without_uv(name: str) -> str | None:
        return None if name == "uv" else real_which(name)

    monkeypatch.setattr(entry.shutil, "which", which_without_uv)
    monkeypatch.setattr(entry, "_machine", lambda: "Test GPU")
    folder = entry.new_entry(tmp_path / "my-entry", name="Test MAPPO")
    printed = capsys.readouterr().out
    assert entry.UV_INSTALL in printed
    assert entry.UV_INSTALL_NOTE in printed
    assert "tdm-season-<N>-test-mappo" in printed
    assert not (folder / "uv.lock").exists()
    _git(folder, "add", "-A")
    _git(folder, "commit", "--quiet", "-m", "Start my entry")
    return folder


def test_starter_files_parse_and_match_the_recipe(starter: Path) -> None:
    version = metadata.version("marl-battlegrounds")
    project = tomllib.loads((starter / "pyproject.toml").read_text())["project"]
    package = f"marl-battlegrounds[training,hydra{{}}]=={version}; sys_platform "
    assert project["dependencies"] == [
        package.format(",cuda13") + "== 'linux'",
        package.format("") + "!= 'linux'",
    ]
    assert (starter / ".python-version").read_text() == "3.14\n"
    assert "runs/" in (starter / ".gitignore").read_text().split()
    module = ast.parse((starter / "my_entry/__init__.py").read_text())
    assert "load" in {n.name for n in module.body if isinstance(n, ast.FunctionDef)}
    settings = entry._read_entry(starter)
    assert (settings.name, settings.repo, settings.load, settings.output) == (
        "Test MAPPO",
        "test-mappo",
        "my_entry:load",
        "runs/entry",
    )
    assert settings.starting_models == ()
    words = shlex.split(settings.train)
    launcher = "uv run python -m marl_battlegrounds.experiments.train"
    assert words[:5] == launcher.split()
    from hydra import compose, initialize_config_module

    with initialize_config_module(
        "marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        recipe = compose("train", overrides=words[5:])
    assert recipe.total_env_steps == 250_675_200 and recipe.num_envs == 512
    assert recipe.retain_all_actors and recipe.output_dir == settings.output
    assert "250,675,200" in (starter / "README.md").read_text()


@pytest.mark.parametrize(
    ("name", "repo"),
    [
        ("My_MAPPO", "my-mappo"),
        ("My Team's MAPPO", "my-team-s-mappo"),
        ("hydra-mappo-2", "hydra-mappo-2"),
        ("a" * 64, "a" * 64),
    ],
)
def test_a_method_name_gives_its_repo_name(name: str, repo: str) -> None:
    assert entry._repo_name(name) == repo


def test_entry_yaml_reads_downloads_and_accepts_1_0_3_files(starter: Path) -> None:
    (starter / "weights.bin").write_bytes(b"small weights")
    (starter / POOL).mkdir(parents=True)
    listed = [
        QMIX,
        {"model": "MARL-BattleGrounds/tdm-season-0-pqn-vdn@100M", "sha256": "a" * 64},
        {"model": POOL, "commit": "b" * 40},
        OUTSIDE,
        {"model": "weights.bin", "sha256": "d" * 64},
        "my_entry",
    ]
    old = {"training_maps": {"training": ["0-41"], "validation": [42]}, "config": "x"}
    _write_entry(starter, starting_models=listed, **old)
    assert entry._read_entry(starter).starting_models == (
        QMIX,
        "MARL-BattleGrounds/tdm-season-0-pqn-vdn@100M",
        POOL,
        OUTSIDE,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"name": "!!!"}, "1 to 64"),
        ({"name": "a" * 65}, "1 to 64"),
        ({"load": "my_entry.load"}, "module:function"),
        ({"seed": 7}, "unknown seed"),
        ({"starting_models": ["a.zip"]}, "not a released name"),
        ({"starting_models": [{"sha256": "a" * 64}]}, "a released name or"),
        ({"starting_models": ["hf://someone/model@main/w.bin"]}, "40-character"),
        ({"starting_models": ["MARL-BattleGrounds/tdm-season-0-qmix@main"]}, "@ must"),
    ],
)
def test_entry_yaml_refuses_bad_settings(
    starter: Path, change: dict[str, object], message: str
) -> None:
    _write_entry(starter, **change)
    with pytest.raises(ValueError, match=message):
        entry._read_entry(starter)


def test_entry_yaml_that_is_not_yaml_is_a_value_error(starter: Path) -> None:
    (starter / "entry.yaml").write_text("name: [never closed\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        entry._read_entry(starter)


PYPI_FILES = "https://files.pythonhosted.org/packages/89/5d/0d"


def _locked(**change: object) -> dict[str, object]:
    package = {
        "name": "marl-battlegrounds",
        "version": "1.0.1",
        "source": {"registry": "https://pypi.org/simple"},
        "sdist": {"url": f"{PYPI_FILES}/marl_battlegrounds-1.0.1.tar.gz"},
        "wheels": [{"url": f"{PYPI_FILES}/marl_battlegrounds-1.0.1-py3-none-any.whl"}],
    }
    return {
        "package": [{"name": "my-entry", "source": {"virtual": "."}}, package | change]
    }


def test_lock_must_install_a_released_package_from_pypi() -> None:
    assert entry._released_package(_locked()) == "1.0.1"
    for version in ("1.0.0", "1.2.0"):
        older_or_newer: dict[str, object] = {
            "version": version,
            "sdist": {"url": f"{PYPI_FILES}/marl_battlegrounds-{version}.tar.gz"},
            "wheels": [],
        }
        assert entry._released_package(_locked(**older_or_newer)) == version
    refused: list[tuple[dict[str, object], str]] = [
        ({"name": "something-else"}, "does not install"),
        ({"source": {"git": "https://github.com/me/fork"}}, "git https://"),
        ({"source": {"editable": "../marl-bgs"}}, "editable ../marl-bgs"),
        ({"source": {"registry": "https://mirror.example/simple"}}, "not from PyPI"),
        (
            {"wheels": [{"url": "https://example.com/marl_battlegrounds-1.0.1-a.whl"}]},
            "example.com",
        ),
        (
            {"wheels": [{"url": f"{PYPI_FILES}/other_package-1.0.1-a.whl"}]},
            "not its file",
        ),
    ]
    for change, message in refused:
        with pytest.raises(entry._StepFailedError, match=message):
            entry._released_package(_locked(**change))


def test_step_4_refuses_a_bad_lock_before_installing(
    starter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def found(name: str) -> str:
        return f"/usr/bin/{name}"

    def no_install(*args: object, **kwargs: object) -> object:
        raise AssertionError("installed despite a bad uv.lock")

    monkeypatch.setattr(entry.shutil, "which", found)
    monkeypatch.setattr(entry, "_run", no_install)
    checker = _checker(starter)
    with pytest.raises(entry._StepFailedError, match=r"No uv\.lock"):
        checker.step_4()
    lines = "\n".join(
        [
            "version = 1",
            "[[package]]",
            'name = "marl-battlegrounds"',
            'version = "1.0.1"',
            'source = { git = "https://github.com/me/fork" }',
        ]
    )
    (starter / "uv.lock").write_text(lines + "\n")
    with pytest.raises(entry._StepFailedError, match="not from PyPI"):
        checker.step_4()


def test_downloads_and_children_run_behind_the_sandbox(
    starter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], Path, dict[str, str], float]] = []
    reply: list[tuple[int | None, str, str]] = [(0, '{"exported": true}', "")]

    def run(
        command: Sequence[str], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> tuple[int | None, str, str]:
        calls.append((list(command), cwd, env, timeout))
        if command[-1].endswith("model.zip"):
            with zipfile.ZipFile(command[-1], "w") as model:
                model.writestr("model.json", "{}")
        return reply[0]

    def sandbox(folder: Path, online: bool) -> list[str]:
        assert folder == starter
        return ["sealed", "online" if online else "offline"]

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def found(name: str) -> str:
        return f"/usr/bin/{name}"

    monkeypatch.setattr(entry.shutil, "which", found)
    monkeypatch.setattr(entry, "_run", run)
    _write_entry(starter, starting_models=[QMIX, OUTSIDE])
    (starter / "uv.lock").write_text(LOCK)
    checker = _checker(starter, sandbox=sandbox)
    assert "2 starting models downloaded" in checker.step_4()
    (sync, _, _, _), online, offline, outside = calls
    assert sync == ["sealed", "online", "uv", "sync", "--locked"]
    for (command, cwd, env, timeout), sealed in (
        (online, "online"),
        (offline, "offline"),
    ):
        assert command[:2] == ["sealed", sealed] and command[-2:] == ["download", QMIX]
        assert cwd == starter and env["HF_HOME"] == str(checker.hf_home)
        assert ("HF_HUB_OFFLINE" in env) == (sealed == "offline")
        assert 0 < timeout <= 3600
    command, cwd, env, _ = outside
    assert command == [sys.executable, "-I", "-c", entry._FETCH_CODE, OUTSIDE]
    assert cwd == starter.parent and "HF_HUB_OFFLINE" not in env
    assert env["HF_HOME"] == str(checker.hf_home)
    calls.clear()
    assert "Exported" in checker.step_7()
    command, _, env, _ = calls[-1]
    assert command[:4] == ["sealed", "offline", "uv", "run"]
    exported = starter / ".entry-check-model.zip"
    assert Path(command[-1]) == exported
    assert (starter.parent / "model.zip").read_bytes() == exported.read_bytes()
    assert env["HF_HUB_OFFLINE"] == "1" and env["HF_HOME"] == str(checker.hf_home)
    assert env == entry.entry_environment(starter, offline=True)
    secret = starter.parent / "secret.txt"
    secret.write_text("hf_not_for_you\n")
    exported.unlink()
    exported.symlink_to(secret)

    def left_a_link(*args: object, **kwargs: object) -> tuple[int, str, str]:
        return 0, '{"exported": true}', ""

    monkeypatch.setattr(entry, "_run", left_a_link)
    with pytest.raises(entry._StepFailedError, match="did not leave a plain file"):
        checker.step_7()
    assert b"hf_not_for_you" not in (starter.parent / "model.zip").read_bytes()
    monkeypatch.setattr(entry, "_run", run)
    reply[0] = (1, "", "no such model")
    with pytest.raises(entry._StepFailedError, match="no such model"):
        checker.fetch(QMIX, time.monotonic() + 60)
    reply[0] = (None, "", "")
    with pytest.raises(entry._StepFailedError, match="more than 1 hour"):
        checker.fetch(QMIX, time.monotonic() + 60)


def test_an_outside_path_with_no_files_fails_its_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    (tmp_path / "weights").mkdir()

    def downloaded(repo: str, **options: object) -> str:
        assert repo == "someone/big-model" and options["token"] is False
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", downloaded)
    for address, finds in ((OUTSIDE, True), (OUTSIDE + "x", False)):
        monkeypatch.setattr(sys, "argv", ["-c", address])
        if finds:
            exec(entry._FETCH_CODE, {})
        else:
            with pytest.raises(SystemExit, match="has no file or folder weightsx"):
                exec(entry._FETCH_CODE, {})


def test_a_run_with_nothing_saved_waits_only_when_stopped(
    starter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def sandbox(folder: Path, online: bool) -> list[str]:
        assert not online
        return ["env", "MARL_BGS_TEST_SEALED=yes"]

    monkeypatch.setattr(entry._Checker, "find_choice", _own_loop)
    _write_entry(starter, train="echo $MARL_BGS_TEST_SEALED > sealed.txt; sleep 30")
    stopped = _checker(starter, sandbox=sandbox)
    assert "then stopped" in stopped.step_5()
    assert stopped.stopped and (starter / "sealed.txt").read_text() == "yes\n"
    stopped.after_training()
    statuses = ["skipped", "skipped", "not measured", "skipped"]
    assert [stopped.steps[n].status for n in (6, 7, 8, 9)] == statuses
    assert "Our run trains for up to 8 hours" in stopped.steps[8].detail

    _write_entry(starter, train="true")
    ended = _checker(starter)
    assert "ended by itself" in ended.step_5() and not ended.stopped
    ended.after_training()
    assert ended.steps[6].status == "failed"
    assert "with nothing saved to enter in runs/entry" in ended.steps[6].detail

    def builtin_without_choice(checker: entry._Checker) -> None:
        checker.builtin_run, checker.choice = True, None

    monkeypatch.setattr(entry._Checker, "find_choice", builtin_without_choice)
    stopped.steps.clear()
    stopped.after_training()
    assert [stopped.steps[n].status for n in (6, 7, 8, 9)] == statuses


def test_a_saved_run_passes_step_8_and_a_step_9_error_fails_step_9(
    starter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (starter / "runs/entry").mkdir(parents=True)
    (starter / "runs/entry/model.zip").write_bytes(b"saved")

    def slow(*args: object, **kwargs: object) -> tuple[None, None, str]:
        return None, None, ""

    monkeypatch.setattr(entry._Checker, "python", slow)
    monkeypatch.setattr(entry._Checker, "find_choice", _own_loop)
    checker = _checker(starter)
    checker.after_training()
    assert checker.steps[6].detail == "Loading took more than 10 minutes."
    assert checker.steps[8].status == "passed"
    assert "runs/entry holds the run's saves" in checker.steps[8].detail

    def done(checker: entry._Checker) -> str:
        return "Done."

    def no_passt(checker: entry._Checker) -> tuple[str, str]:
        raise RuntimeError("install passt")

    monkeypatch.setattr(entry._Checker, "step_6", done)
    monkeypatch.setattr(entry._Checker, "step_7", done)
    monkeypatch.setattr(entry._Checker, "step_9", no_passt)
    checker = _checker(starter)
    checker.after_training()
    assert checker.steps[9].status == "failed"
    assert checker.steps[9].detail == "RuntimeError: install passt"


@pytest.mark.parametrize(
    ("device", "peak", "status"),
    [
        ("cpu", None, "not measured"),
        ("NVIDIA GeForce RTX 4090", 10**9, "not measured"),
        ("NVIDIA GeForce RTX 5090", None, "not measured"),
        ("NVIDIA GeForce RTX 5090", 10**9, "passed"),
        ("NVIDIA GeForce RTX 5090", 11 * 10**9, "failed"),
    ],
)
def test_step_9_decides_only_on_an_rtx_5090(
    starter: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    peak: int | None,
    status: str,
) -> None:
    reading = {
        "reference": "model.zip",
        "device": device,
        "on_gpu": device != "cpu",
        "games": 2048,
        "game_steps": 600_000,
        "seconds": 10.0,
        "warmup_seconds": 30.0,
        "peak_memory_bytes": peak,
    }

    def played(
        command: Sequence[str], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> tuple[int, str, str]:
        export = str(starter.parent / "model.zip")
        assert list(command) == [sys.executable, "-I", "-c", entry._PLAY_CODE, export]
        assert "--xla_gpu_deterministic_ops=true" in env["XLA_FLAGS"]
        return 0, json.dumps(reading), ""

    monkeypatch.setattr(entry, "_run", played)
    assert _checker(starter).step_9()[0] == status


def test_steps_sort_into_must_pass_and_measured_here() -> None:
    def result(failed: int) -> entry.EntryCheck:
        steps = tuple(
            entry.CheckStep(n, name, group, "failed" if n == failed else "passed", "")
            for n, name, group in entry._STEPS
        )
        return entry.EntryCheck("repo", "0" * 40, "GPU", "now", "1", steps)

    check = result(failed=9)
    assert [s.number for s in check.must_pass] == [1, 2, 3, 4, 5, 6, 7]
    assert [s.number for s in check.measured_here] == [8, 9]
    assert check.passed and json.loads(json.dumps(check.to_dict()))["passed"]
    assert result(failed=8).passed and not result(failed=3).passed


def _stub_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    def later_steps(checker: entry._Checker) -> None:
        checker.record(6, "passed", "Loaded.")
        checker.record(7, "passed", "Exported.")
        checker.record(8, "not measured", "Nothing saved yet.")
        checker.record(9, "not measured", "No GPU here.")

    def installed(checker: entry._Checker) -> str:
        return "Installed."

    monkeypatch.setattr(entry._Checker, "step_4", installed)
    monkeypatch.setattr(entry._Checker, "step_5", installed)
    monkeypatch.setattr(entry._Checker, "after_training", later_steps)


def test_check_and_submit(
    starter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    _git(empty, "init", "--quiet")
    with pytest.raises(ValueError, match="no commit yet"):
        entry.submit_entry(empty, open_browser=False)
    (starter / "notes.txt").write_text("not committed\n")
    with pytest.raises(ValueError, match="uncommitted"):
        entry.submit_entry(starter, open_browser=False)
    dirty = entry.check_entry(starter)
    assert not dirty.passed and dirty.record is None
    assert [s.status for s in dirty.steps] == ["failed"] + ["skipped"] * 8
    (starter / "notes.txt").unlink()

    _stub_steps(monkeypatch)
    check = entry.check_entry(starter)
    commit = _git(starter, "rev-parse", "HEAD")
    assert check.passed and check.commit == commit
    assert check.steps[2].detail == (
        "Method 'Test MAPPO' (Hugging Face repo tdm-season-<N>-test-mappo); "
        "load my_entry:load."
    )
    record = Path(str(check.record))
    assert record == starter / f".git/marl-bgs-entry-check/{commit}.json"
    assert json.loads(record.read_text())["passed"] is True
    with pytest.raises(ValueError, match="This is the starter"):
        entry.submit_entry(starter, open_browser=False)

    _write_entry(starter, train=entry._STARTER_TRAIN + " seed=7")
    _git(starter, "commit", "--quiet", "-am", "Change the method")
    commit = _git(starter, "rev-parse", "HEAD")
    assert entry.check_entry(starter).passed
    with pytest.raises(ValueError, match="no remote branch"):
        entry.submit_entry(starter, open_browser=False)
    _git(tmp_path, "init", "--quiet", "--bare", "remote.git")
    _git(starter, "remote", "add", "origin", str(tmp_path / "remote.git"))
    _git(starter, "push", "--quiet", "origin", "HEAD:refs/heads/main")
    link = urllib.parse.urlsplit(entry.submit_entry(starter, open_browser=False))
    values = dict(urllib.parse.parse_qsl(link.query))
    assert link.geturl().startswith(entry.FORM_URL + "?")
    assert set(values) == {"template", "title", *_form_text_fields()}
    assert (values["template"], values["title"]) == (entry.FORM_TEMPLATE, "Test MAPPO")
    assert values["commit"] == commit

    (starter / "README.md").write_text("Changed.\n")
    _git(starter, "commit", "--quiet", "-am", "Change the README")
    _git(starter, "push", "--quiet", "origin", "HEAD:refs/heads/main")
    with pytest.raises(ValueError, match="no passing check"):
        entry.submit_entry(starter, open_browser=False)
    _git(starter, "remote", "set-url", "origin", "https://me:FAKE@github.com/t/e.git")
    assert entry._remote_url(starter, commit) == "https://github.com/t/e"


def test_the_choice_is_read_by_a_sealed_child(
    starter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def answered(*args: object, **kwargs: object) -> tuple[int, dict[str, Any], str]:
        assert args[1:3] == (entry._CHOICE_CODE, str(starter / "runs/entry"))
        return 0, {"builtin": True, "choice": "/clone/runs/entry/actors/a"}, ""

    monkeypatch.setattr(entry._Checker, "python", answered)
    checker = _checker(starter)
    checker.find_choice()
    assert checker.builtin_run and checker.choice == Path("/clone/runs/entry/actors/a")
    monkeypatch.setattr(sys, "argv", ["-c", str(starter / "runs/entry")])
    exec(entry._CHOICE_CODE, {})
    assert json.loads(capsys.readouterr().out) == {"builtin": False, "choice": None}


def test_a_committed_link_out_of_the_repository_fails_step_2(
    starter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_steps(monkeypatch)
    (starter / "inside").symlink_to("README.md")
    _git(starter, "add", "inside")
    _git(starter, "commit", "--quiet", "-m", "A link inside")
    assert entry.check_entry(starter).passed
    (starter / "entry.yaml").unlink()
    (starter / "entry.yaml").symlink_to("/etc/hostname")
    _git(starter, "commit", "--quiet", "-am", "A link outside")
    check = entry.check_entry(starter)
    assert check.steps[1].status == "failed"
    assert "point outside the repository: entry.yaml" in check.steps[1].detail


def test_a_kept_folder_keeps_the_clone(
    starter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_steps(monkeypatch)
    kept = tmp_path / "official" / "run"
    assert entry.check_entry(starter, keep_folder=kept).passed
    commit = _git(starter, "rev-parse", "HEAD")
    assert _git(kept / "entry", "rev-parse", "HEAD") == commit
    assert (kept / "entry/entry.yaml").is_file()
    with pytest.raises(ValueError, match="already exists"):
        entry.check_entry(starter, keep_folder=kept)


@pytest.mark.parametrize("visible", ["", "-1", "-1,0"])
def test_the_machine_label_names_no_hidden_gpu(
    visible: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert entry._machine() == "No GPU visible"


def test_play_limits_plays_on_the_cpu() -> None:
    reading = entry.play_limits("random", games=2, batches=1)
    assert (reading.on_gpu, reading.games, reading.device) == (False, 2, "cpu")
    assert 2 <= reading.game_steps <= 600 and reading.seconds > 0
    assert reading.within_limits is None
    assert json.loads(json.dumps(reading.to_dict()))["games"] == 2
    with pytest.raises(ValueError, match="even"):
        entry.play_limits("random", games=3)


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGHUP])
def test_a_stop_signal_interrupts_the_check_and_kills_its_child_group(
    number: signal.Signals,
) -> None:
    before = signal.getsignal(number)
    child = subprocess.Popen(["sleep", "60"], start_new_session=True, text=True)
    with (
        pytest.raises(KeyboardInterrupt),
        entry._stop_signals_interrupt(),
        entry._group_stopped_on_error(child),
    ):
        os.kill(os.getpid(), number)
        child.wait()
    assert child.poll() is not None
    assert signal.getsignal(number) is before
