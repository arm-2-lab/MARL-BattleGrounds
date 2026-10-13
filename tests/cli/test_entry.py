"""Check the leaderboard entry commands in marl_battlegrounds.entry.

new_entry writes a starter whose files parse: pyproject.toml pins the installed
marl-battlegrounds version with the CUDA 13 build on Linux only,
.python-version is 3.14, .gitignore keeps runs/ out, my_entry defines load,
and entry.yaml passes the same validation entry check uses. Its train line
composes through the packaged Hydra configs into the released Recurrent MAPPO
recipe (250,675,200 steps, 512 games, every actor kept), validating on the
maps entry.yaml declares. Without uv it prints the uv install line, then the
note that an installer ERROR about ~/.config still leaves uv installed, and
stops before locking. entry.yaml refuses maps outside 0-46 and models without a
SHA-256.

Step 4 accepts a uv.lock only when every marl-battlegrounds in it is a release
from PyPI's index, with each file that version's own file on PyPI's file
server; any released version passes, older or newer (1.0.0, 1.0.1, 1.2.0). It
refuses a missing package, a Git, folder or other-index source, and a file
elsewhere or of another package. A missing or bad uv.lock fails step 4 before
anything is installed.

check_entry sorts steps 1-7 as must pass and 8-9 as measured here: a failed
measured step never fails the check, a failed must-pass step does. A dirty
tree fails step 1, skips the rest and saves no record. With install, training
and the later steps stubbed (no uv, no training), a clean commit really
clones, validates entry.yaml and saves its record under .git, where
submit_entry reads it.

submit_entry refuses uncommitted changes, a commit on no remote branch and a
commit without a passing check. A user name or token inside the remote's
address never reaches the link. Its link fills exactly the form's text fields
other than team (read from .github/ISSUE_TEMPLATE/leaderboard_submission.yml)
plus the title; a link over the length limit falls back to the title alone
and prints the values. play_limits plays 2 self-play games on the CPU through
the real evaluate path and reports them as not a GPU measurement.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import ast
import json
import shlex
import shutil
import subprocess
import tomllib
import urllib.parse
from importlib import metadata
from pathlib import Path

import pytest
import yaml

from marl_battlegrounds import entry

FORM = Path(__file__).parents[2] / ".github/ISSUE_TEMPLATE/leaderboard_submission.yml"


def _git(folder: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=folder, capture_output=True, text=True, check=True
    ).stdout.strip()


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
    folder = entry.new_entry(tmp_path / "my-entry", name="Test Team MAPPO")
    printed = capsys.readouterr().out
    assert entry.UV_INSTALL in printed
    assert entry.UV_INSTALL_NOTE in printed
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
    assert (settings.name, settings.load, settings.output) == (
        "Test Team MAPPO",
        "my_entry:load",
        "runs/entry",
    )
    assert settings.training_maps == tuple(range(42))
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
    assert tuple(recipe.validation_maps) == settings.validation_maps
    assert "250,675,200" in (starter / "README.md").read_text()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"training_maps": {"training": ["0-47"], "validation": []}}, "Rule 3.4"),
        ({"starting_models": [{"model": "a.zip", "sha256": "abc"}]}, "sha256"),
        ({"load": "my_entry.load"}, "module:function"),
    ],
)
def test_entry_yaml_refuses_bad_settings(
    starter: Path, change: dict[str, object], message: str
) -> None:
    path = starter / "entry.yaml"
    path.write_text(yaml.safe_dump({**yaml.safe_load(path.read_text()), **change}))
    with pytest.raises(ValueError, match=message):
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
    checker = entry._Checker(str(starter), None, starter.parent, 1.0, None)
    checker.clone, checker.entry = starter, entry._read_entry(starter)
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


def test_steps_sort_into_must_pass_and_measured_here() -> None:
    def result(failed: int) -> entry.EntryCheck:
        steps = tuple(
            entry.CheckStep(n, rules, name, group, "passed", "")
            if n != failed
            else entry.CheckStep(n, rules, name, group, "failed", "")
            for n, rules, name, group in entry._STEPS
        )
        return entry.EntryCheck("repo", "0" * 40, "GPU", "now", "1", steps)

    check = result(failed=9)
    assert [s.number for s in check.must_pass] == [1, 2, 3, 4, 5, 6, 7]
    assert [s.number for s in check.measured_here] == [8, 9]
    assert check.passed and json.loads(json.dumps(check.to_dict()))["passed"]
    assert result(failed=8).passed and not result(failed=3).passed


def test_check_and_submit(
    starter: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (starter / "notes.txt").write_text("not committed\n")
    with pytest.raises(ValueError, match="uncommitted"):
        entry.submit_entry(starter, open_browser=False)
    dirty = entry.check_entry(starter)
    assert not dirty.passed and dirty.record is None
    assert [s.status for s in dirty.steps] == ["failed"] + ["skipped"] * 8
    (starter / "notes.txt").unlink()

    def later_steps(checker: entry._Checker, machine: str) -> None:
        checker.record(6, "passed", "Loaded.")
        checker.record(7, "passed", "Exported.")
        checker.record(8, "failed", f"On {machine}: no choice yet.")
        checker.record(9, "not measured", "No GPU here.")

    def installed(checker: entry._Checker) -> str:
        return "Installed."

    monkeypatch.setattr(entry._Checker, "step_4", installed)
    monkeypatch.setattr(entry._Checker, "step_5", installed)
    monkeypatch.setattr(entry._Checker, "after_training", later_steps)
    check = entry.check_entry(starter)
    commit = _git(starter, "rev-parse", "HEAD")
    assert check.passed and check.commit == commit
    assert check.steps[2].detail == "Name 'Test Team MAPPO'; load my_entry:load."
    record = Path(str(check.record))
    assert record == starter / f".git/marl-bgs-entry-check/{commit}.json"
    assert json.loads(record.read_text())["passed"] is True

    with pytest.raises(ValueError, match="no remote branch"):
        entry.submit_entry(starter, open_browser=False)
    _git(tmp_path, "init", "--quiet", "--bare", "remote.git")
    _git(starter, "remote", "add", "origin", str(tmp_path / "remote.git"))
    _git(starter, "push", "--quiet", "origin", "HEAD:refs/heads/main")
    link = urllib.parse.urlsplit(entry.submit_entry(starter, open_browser=False))
    values = dict(urllib.parse.parse_qsl(link.query))
    assert link.geturl().startswith(entry.FORM_URL + "?")
    assert set(values) == {"template", "title", *entry.FORM_FIELDS}
    assert values["template"] == entry.FORM_TEMPLATE
    assert (values["commit"], values["lock-file"]) == (commit, "uv.lock")
    assert values["starting-models"] == "none"
    assert "Training maps: 0-41" in values["config-and-maps"]

    monkeypatch.setattr(entry, "MAX_LINK_LENGTH", 100)
    capsys.readouterr()
    short = urllib.parse.urlsplit(entry.submit_entry(starter, open_browser=False))
    assert set(dict(urllib.parse.parse_qsl(short.query))) == {"template", "title"}
    assert f"train-command:\n{values['train-command']}" in capsys.readouterr().out

    (starter / "README.md").write_text("Changed.\n")
    _git(starter, "commit", "--quiet", "-am", "Change the README")
    _git(starter, "push", "--quiet", "origin", "HEAD:refs/heads/main")
    with pytest.raises(ValueError, match="no passing check"):
        entry.submit_entry(starter, open_browser=False)
    _git(starter, "remote", "set-url", "origin", "https://me:FAKE@github.com/t/e.git")
    assert entry._remote_url(starter, commit) == "https://github.com/t/e"


def test_form_fields_are_the_form_files_text_fields() -> None:
    body = yaml.safe_load(FORM.read_text())["body"]
    text = {f["id"] for f in body if f["type"] in ("input", "textarea")}
    assert set(entry.FORM_FIELDS) == text - {"team"}


def test_play_limits_plays_on_the_cpu() -> None:
    reading = entry.play_limits("random", games=2, batches=1)
    assert (reading.on_gpu, reading.games, reading.device) == (False, 2, "cpu")
    assert 2 <= reading.game_steps <= 600 and reading.seconds > 0
    assert reading.within_limits is None
    assert json.loads(json.dumps(reading.to_dict()))["games"] == 2
    with pytest.raises(ValueError, match="even"):
        entry.play_limits("random", games=3)
