"""Enter the leaderboard: start an entry, check it the way we will, and submit it.

This module owns the three entry commands. The command line only parses and
calls these functions (``python -m marl_battlegrounds entry new|check|submit``):

- ``new_entry`` writes a starter repository that trains the released Recurrent
  MAPPO recipe, runs ``git init`` and locks it with ``uv lock``.
- ``check_entry`` checks one commit the way we check every entry, in nine
  numbered steps. Steps 1 to 7 must pass on any machine and block
  ``entry submit``. Steps 8 and 9 are measured here but decided on our
  RTX 5090, so they never block. The result is saved under the repository's
  ``.git`` folder, so it is never committed. Maintainers run the same function
  for the intake test and for the official run.
- ``submit_entry`` refuses unfinished work and the unchanged starter, then
  opens the GitHub issue form with the method name, repository and commit
  filled in.
- ``play_limits`` measures the play limits: the system plays itself and we
  time it.

Who owns what. The rules live in ``docs/tournament_rules.md``; the tools state
facts and cite no rule numbers, so a rule can change without making a message
here wrong. Existing owners do the real work: the entry's own train command
trains, ``load_factory`` (``module:function`` loading) loads, ``export_model``
exports, ``models download`` fetches released models and training-opponent
pools, ``huggingface_hub`` fetches outside ``hf://`` models (``released_models``
owns released names and parses ``hf://`` addresses),
``training.current_choice`` names a built-in run's choice, and ``evaluate``
plays the games. Researcher code (install, train, load, export, play) always
runs in child processes inside the entry's own locked environment, never in
the caller's Python. This module is POSIX only (it stops process groups), like
the rest of the package's run files.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.parse
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, cast

from marl_battlegrounds._browser import open_in_browser
from marl_battlegrounds.released_models import (
    RELEASE_ORGANIZATION,
    _parse_address,  # pyright: ignore[reportPrivateUsage]
    _pool,  # pyright: ignore[reportPrivateUsage]
    _typed_name,  # pyright: ignore[reportPrivateUsage]
)

FORM_URL = "https://github.com/arm-2-lab/MARL-BattleGrounds/issues/new"
"""The GitHub page that opens a new issue in the MARL-BattleGrounds repository."""

FORM_TEMPLATE = "leaderboard_submission.yml"
"""The issue form's file name in ``.github/ISSUE_TEMPLATE``."""

OFFICIAL_TRAIN_MINUTES = 480.0
"""How long our official run trains each entry, in minutes (8 hours)."""

MIN_GAME_STEPS_PER_SECOND = 5000
"""The slowest allowed playing speed, in game steps per second (one tick of one
game)."""

MAX_GPU_MEMORY_BYTES = 10 * 10**9
"""The most GPU memory a system may use while playing, in bytes (10 GB, with
1 GB = 10**9 bytes)."""

PLAY_MAPS = (42, 43, 44, 45, 46)
"""The validation maps ``play_limits`` plays on."""

ENTRY_TYPES = ("learned", "scripted", "hybrid", "language model")
"""The entry types the table shows, as written in ``entry.yaml`` (any letter case)."""

UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
"""The one-line uv install for Linux and macOS, from uv's own guide."""

UV_INSTALL_NOTE = (
    "If the installer ends with an ERROR about ~/.config, uv is still installed: "
    "open a new terminal."
)
"""Printed under ``UV_INSTALL``: uv's installer puts uv in place, then can fail on
a later step when ``~/.config`` is not the user's to write (often after an
earlier ``sudo``)."""

_RECORD_FOLDER = "marl-bgs-entry-check"
_ENTRY_FILE = "entry.yaml"
_CUDA_JAX = frozenset({"jax-cuda13-plugin", "jax-cuda12-plugin"})
_PYPI_INDEX = "https://pypi.org/simple"
_PYPI_FILES = "https://files.pythonhosted.org/"
_INSTALL_SECONDS = 3600.0
_STOP_GRACE_SECONDS = 60.0
_LOAD_SECONDS = 600.0
_EXPORT_SECONDS = 600.0
_PLAY_SECONDS = 1800.0
_DETERMINISTIC = "--xla_gpu_deterministic_ops=true"
_SETTINGS = ("name", "train", "output", "load", "entry_type", "starting_models")
# 1.0.3's entry.yaml also had these two; they are accepted and ignored.
_OLD_SETTINGS = ("training_maps", "config")
_MAX_REPO_NAME = 64
_REPO_PATTERN = "tdm-season-<N>-{}"
_INSTALL_LIMIT = "Installing and downloading took more than 1 hour."
_EXPORT_FILE = ".entry-check-model.zip"
_HF_FOLDER = ".entry-check-hf"

Sandbox = Callable[[Path, bool], Sequence[str]]
"""A maintainers' seal: ``sandbox(clone, online)`` returns a command prefix.

``check_entry`` puts the prefix (a list of strings, for example a ``bwrap``
command) in front of every child that runs the entry's code. ``clone`` is the
clone's folder, the only folder such a child writes. ``online`` is True in
step 4 (the install and the released-model downloads need the network) and
False in steps 5 to 7. The prefix must keep the environment the check gives
the child, keep the child in the check's process group (no new session) and
let SIGINT reach it, or step 5 cannot stop the run politely. Outside ``hf://``
models are not downloaded behind it: the check's own Python fetches them.
"""

StepGroup = Literal["must pass", "measured here"]
"""Must-pass steps block ``entry submit``; measured-here steps never do."""

StepStatus = Literal["passed", "failed", "skipped", "not measured"]
"""What happened to one step. "skipped" means it was not run; its detail says why."""

_STEPS: tuple[tuple[int, str, StepGroup], ...] = (
    (1, "The tree is clean and the commit is recorded", "must pass"),
    (2, "A fresh clone of the commit", "must pass"),
    (3, "entry.yaml is present and valid", "must pass"),
    (4, "Install from uv.lock and fetch starting models", "must pass"),
    (5, "The train command runs with downloads off, then stops", "must pass"),
    (6, "The load function returns a Policy or System", "must pass"),
    (7, "Export to a portable model file", "must pass"),
    (8, "Something to enter saved within the training time", "measured here"),
    (9, "Play limits", "measured here"),
)


@dataclass(frozen=True)
class CheckStep:
    """The result of one numbered step of ``check_entry``.

    Attributes
    ----------
    number : int
        The step number, 1 to 9, as in ``check_entry``'s list and the lines it
        prints.
    title : str
        What the step checks, in a few words.
    group : {"must pass", "measured here"}
        Must-pass steps (1 to 7) block ``entry submit`` when they fail.
        Measured-here steps (8 and 9) are decided on our RTX 5090 and never
        block.
    status : {"passed", "failed", "skipped", "not measured"}
        "skipped" means the step did not run, for example because an earlier
        step failed or nothing was saved to check yet. "not measured" means
        this machine or this short run cannot measure it, for example step 9
        without an RTX 5090.
    detail : str
        A plain-English explanation: what was found, or what to fix.
    """

    number: int
    title: str
    group: StepGroup
    status: StepStatus
    detail: str


@dataclass(frozen=True)
class PlayLimits:
    """One reading of the play limits: a system playing itself, timed.

    Attributes
    ----------
    reference : str
        The system that played, as given to ``play_limits``.
    device : str
        JAX's name for the device that played, for example
        ``"NVIDIA GeForce RTX 5090"`` or ``"cpu"``.
    on_gpu : bool
        True when the games ran on a GPU. Only a GPU reading measures the
        play limits.
    games : int
        Games in the timed call (the warm-up batch is not counted).
    game_steps : int
        Ticks summed over the timed games. One game step is one tick of one
        game.
    seconds : float
        Wall time of the timed call, in seconds.
    warmup_seconds : float
        Wall time of the warm-up batch, in seconds. It includes compiling.
    peak_memory_bytes : int or None
        The most device memory JAX had in use at once in this process, in
        bytes. None when the device does not report it (the CPU does not).
    """

    reference: str
    device: str
    on_gpu: bool
    games: int
    game_steps: int
    seconds: float
    warmup_seconds: float
    peak_memory_bytes: int | None

    @property
    def game_steps_per_second(self) -> float:
        """Game steps per second of the timed call (0.0 if it took no time)."""
        return self.game_steps / self.seconds if self.seconds > 0 else 0.0

    @property
    def within_limits(self) -> bool | None:
        """True when both play limits hold, False when one fails.

        The limits are at least MIN_GAME_STEPS_PER_SECOND and at most
        MAX_GPU_MEMORY_BYTES. None when this is not a GPU reading or the memory
        is unknown, because then the limits were not measured.
        """
        if not self.on_gpu or self.peak_memory_bytes is None:
            return None
        return (
            self.game_steps_per_second >= MIN_GAME_STEPS_PER_SECOND
            and self.peak_memory_bytes <= MAX_GPU_MEMORY_BYTES
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the reading as plain JSON values, with the speed included."""
        return {
            "reference": self.reference,
            "device": self.device,
            "on_gpu": self.on_gpu,
            "games": self.games,
            "game_steps": self.game_steps,
            "seconds": self.seconds,
            "warmup_seconds": self.warmup_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "game_steps_per_second": self.game_steps_per_second,
        }


@dataclass(frozen=True)
class EntryCheck:
    """The result of ``check_entry``: every step, in two groups.

    Attributes
    ----------
    repository : str
        The folder or URL that was checked.
    commit : str or None
        The full commit SHA that was checked. None when step 1 found no commit
        (none yet, or the folder is not in a Git repository).
    machine : str
        The name ``nvidia-smi`` gives the first GPU this process may use, "No
        GPU visible" when ``CUDA_VISIBLE_DEVICES`` hides them all, or "No
        NVIDIA GPU found". It is only recorded; no step is judged by it.
    checked_at : str
        When the check finished, in UTC, ISO 8601.
    package_version : str
        The marl-battlegrounds version that ran the check.
    steps : tuple of CheckStep
        Steps 1 to 9, in order. Steps that did not run are "skipped".
    play : PlayLimits or None
        Step 9's GPU reading, or None when step 9 did not measure on a GPU.
    record : str or None
        Where the result was saved as JSON. None when nothing was saved
        because step 1 failed (no commit, a tree that is not clean, or a
        folder outside a Git repository).
    """

    repository: str
    commit: str | None
    machine: str
    checked_at: str
    package_version: str
    steps: tuple[CheckStep, ...]
    play: PlayLimits | None = None
    record: str | None = None

    @property
    def must_pass(self) -> tuple[CheckStep, ...]:
        """Steps 1 to 7: they must pass on any machine."""
        return tuple(step for step in self.steps if step.group == "must pass")

    @property
    def measured_here(self) -> tuple[CheckStep, ...]:
        """Steps 8 and 9: measured on this machine, decided on our RTX 5090."""
        return tuple(step for step in self.steps if step.group == "measured here")

    @property
    def passed(self) -> bool:
        """True when no must-pass step failed. Measured-here steps never count."""
        return not any(step.status == "failed" for step in self.must_pass)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as plain JSON values; ``entry submit`` reads it back."""
        return {
            "repository": self.repository,
            "commit": self.commit,
            "machine": self.machine,
            "checked_at": self.checked_at,
            "package_version": self.package_version,
            "passed": self.passed,
            "steps": [
                {
                    "step": step.number,
                    "title": step.title,
                    "group": step.group,
                    "status": step.status,
                    "detail": step.detail,
                }
                for step in self.steps
            ],
            "play": None if self.play is None else self.play.to_dict(),
        }


@dataclass(frozen=True)
class _Entry:
    """The checked contents of one ``entry.yaml`` (see _read_entry).

    ``repo`` is the method name's part of its Hugging Face repo name (see
    _repo_name). ``starting_models`` holds the downloads in their listed order.
    """

    name: str
    repo: str
    train: str
    output: str
    load: str
    entry_type: str
    starting_models: tuple[str, ...]


class _StepFailedError(RuntimeError):
    """A Git command or a check step failed; the message says what to fix.

    Inside ``check_entry`` the message becomes the step's detail. Elsewhere it
    reaches the caller as a RuntimeError.
    """


def _package_version() -> str:
    """Return the installed marl-battlegrounds version, for example "1.0.1"."""
    return metadata.version("marl-battlegrounds")


def _tail(text: str, lines: int = 20) -> str:
    """Return the last ``lines`` lines of a program's output, trimmed."""
    return "\n".join(text.strip().splitlines()[-lines:])


def _last_json(out: str) -> dict[str, Any] | None:
    """Return the JSON object on the last line of a child's output, or None."""
    lines = out.strip().splitlines()
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(lines[-1]) if lines else None
    return None


def _git(folder: Path, *args: str) -> str:
    """Run one Git command in ``folder`` and return its trimmed output.

    Raises _StepFailedError (a RuntimeError) with Git's own message when Git
    exits with an error. Uses the network only when ``args`` asks for it
    (``clone`` of a URL).
    """
    done = subprocess.run(
        ["git", *args], cwd=folder, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise _StepFailedError(f"git {args[0]} failed: {_tail(done.stderr, 5)}")
    return done.stdout.strip()


def _stop_group(process: subprocess.Popen[str], grace: float) -> bool:
    """Stop a child's process group: a stop signal, ``grace`` seconds, then a kill.

    Returns True when the group stopped within the grace time, False when it
    had to be killed. With ``grace=0`` the kill comes at once if it is still
    running. A group that already ended is fine.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=grace)
        return True
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return False


@contextlib.contextmanager
def _group_stopped_on_error(process: subprocess.Popen[str]) -> Generator[None]:
    """Kill a child's process group at once if the block stops with any error.

    The children run in their own process groups, so nothing else stops them
    when the check itself is interrupted (Ctrl-C, or a stop signal turned into
    KeyboardInterrupt by ``_stop_signals_interrupt``). The error is raised again.
    """
    try:
        yield
    except BaseException:
        _stop_group(process, 0)
        raise


@contextlib.contextmanager
def _stop_signals_interrupt() -> Generator[None]:
    """Make SIGTERM and SIGHUP raise KeyboardInterrupt, as Ctrl-C does, for a while.

    By default these signals end Python at once, skipping every cleanup, so a
    training run the check started (in its own process group) kept running
    after ``timeout``, ``kill`` or a closed terminal. The previous handlers come
    back when the block ends. Python lets only the main thread set handlers;
    elsewhere nothing changes.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(number: int, frame: object) -> None:
        """Raise KeyboardInterrupt in the main thread for a stop signal."""
        del number, frame
        raise KeyboardInterrupt

    previous = {
        number: signal.signal(number, interrupt)
        for number in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _run(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout: float
) -> tuple[int | None, str, str]:
    """Run a child in its own process group and return (exit code, stdout, stderr).

    The exit code is None when the child ran past ``timeout`` seconds; its whole
    process group is then killed, as it is when this call is interrupted. Output
    is collected as text.
    """
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    with _group_stopped_on_error(process):
        try:
            out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _stop_group(process, 0)
            out, err = process.communicate()
            return None, out, err
    return process.returncode, out, err


def _repo_name(name: str) -> str:
    """Return the method name's part of the entry's Hugging Face repo name.

    The repo is ``tdm-season-<N>-<part>``, where ``<N>`` is the season the
    entry is sent to. The part is ``name`` in lowercase, with each run of
    characters other than ``a`` to ``z`` and ``0`` to ``9`` turned into one
    hyphen and the hyphens at both ends removed: "My_MAPPO" gives "my-mappo".
    ``entry new`` and step 3 print it; the maintainers refuse a part another
    entry already uses this season.

    Raises
    ------
    ValueError
        The part is empty or longer than 64 characters.
    """
    part = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not part or len(part) > _MAX_REPO_NAME:
        raise ValueError(
            f"The method name {name!r} gives the repo name {part!r}. Choose a "
            f"name that gives 1 to {_MAX_REPO_NAME} letters, digits and hyphens."
        )
    return part


def _read_entry(folder: Path) -> _Entry:
    """Read and check ``folder/entry.yaml``; raise ValueError at the first problem.

    Parameters
    ----------
    folder : Path
        The entry's top folder. A starting model that is neither a released
        name nor an ``hf://`` address but names a path inside it is a
        repository file.

    Returns
    -------
    _Entry
        The settings, trimmed, with the method name's repo part.
        ``starting_models`` keeps the downloads only, in the listed order.

    Raises
    ------
    FileNotFoundError
        There is no ``entry.yaml``.
    ValueError
        The file is not valid YAML or not ``key: value`` lines; a key is
        missing or unknown; ``name``, ``train``, ``output``, ``load`` or
        ``entry_type`` is empty or not text; the name gives no repo name (see
        _repo_name); ``output`` is not a folder inside the repository;
        ``load`` is not ``module:function``; ``entry_type`` is not one of
        ENTRY_TYPES; ``starting_models`` is not a list; or a starting model is
        not a released name, an ``hf://`` address with a full commit, or a path
        inside the repository.

    Notes
    -----
    The keys are ``name`` (the method name), ``train``, ``output``,
    ``load``, ``entry_type`` and ``starting_models``. 1.0.3's
    ``training_maps`` and ``config`` are accepted and ignored. Each starting
    model is one string: a released name, as ``models download`` takes it
    (``MARL-BattleGrounds/<repo>@final`` or a training-opponent pool such as
    ``MARL-BattleGrounds/tdm-season-0-results/training-opponents``), or an
    exact address, ``hf://<owner>/<repo>@<40-character commit>/<path>``.
    1.0.3's mapping form, ``{model: ..., sha256: ...}`` (``commit:`` for a
    pool), is read by ``model`` alone. A path inside the repository needs no
    line, because the commit fixes its bytes; one that is listed is left
    out. Needs PyYAML, which the hydra extra installs.
    """
    import yaml

    path = folder / _ENTRY_FILE
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{_ENTRY_FILE} is not valid YAML: {error}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{_ENTRY_FILE} must hold the settings as key: value lines")
    data = cast(dict[str, object], raw)
    missing = [key for key in _SETTINGS if key not in data]
    known = (*_SETTINGS, *_OLD_SETTINGS)
    extra = sorted(str(key) for key in data if key not in known)
    if missing or extra:
        raise ValueError(
            f"{_ENTRY_FILE} needs the keys {', '.join(_SETTINGS)}; missing "
            f"{', '.join(missing) or 'none'}, unknown {', '.join(extra) or 'none'}"
        )
    text: dict[str, str] = {}
    for key in ("name", "train", "output", "load", "entry_type"):
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{_ENTRY_FILE}: {key} is empty or not text")
        text[key] = value.strip()
    try:
        repo = _repo_name(text["name"])
    except ValueError as error:
        raise ValueError(f"{_ENTRY_FILE}: {error}") from None
    output = Path(text["output"])
    if output.is_absolute() or ".." in output.parts:
        raise ValueError(
            f"{_ENTRY_FILE}: output must be a folder inside the repository"
        )
    if not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", text["load"]):
        raise ValueError(
            f"{_ENTRY_FILE}: load must be module:function, like my_entry:load"
        )
    if text["entry_type"].lower() not in ENTRY_TYPES:
        raise ValueError(
            f"{_ENTRY_FILE}: entry_type must be Learned, Scripted, Hybrid or "
            "Language Model"
        )
    listed = data["starting_models"]
    if not isinstance(listed, list):
        raise ValueError(f"{_ENTRY_FILE}: starting_models must be a list ([] for none)")
    top = folder.resolve()
    models: list[str] = []
    for item in cast(list[object], listed):
        model: object = (
            cast(dict[str, object], item).get("model")
            if isinstance(item, dict)
            else item
        )
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"{_ENTRY_FILE}: write each starting model as one line of text, "
                f"a released name or an hf:// address; found {item!r}"
            )
        model = model.strip()
        try:
            if model.startswith("hf://"):
                _parse_address(model)
            elif _typed_name(model) is None and _pool(model) is None:
                inside = (folder / model).resolve()
                if inside.is_relative_to(top) and inside.exists():
                    continue
                raise ValueError(
                    f"{model!r} is not a released name, an hf:// address or a "
                    "path inside this repository"
                )
        except ValueError as error:
            raise ValueError(f"{_ENTRY_FILE}: starting_models: {error}") from None
        models.append(model)
    return _Entry(
        name=text["name"],
        repo=repo,
        train=text["train"],
        output=output.as_posix(),
        load=text["load"],
        entry_type=text["entry_type"],
        starting_models=tuple(models),
    )


_STARTER_TRAIN = (
    "uv run python -m marl_battlegrounds.experiments.train method=mappo "
    "pool=scripted 'validation_opponents=[tdm-alpha,tdm-beta,tdm-gamma]' "
    "output_dir=runs/entry"
)


def _starter_files(name: str, project: str, version: str) -> dict[str, str]:
    """Return the starter's files, by path relative to the entry folder.

    ``name`` is the method name, ``project`` the Python project name and
    ``version`` the marl-battlegrounds version the starter depends on.
    """
    train = json.dumps(_STARTER_TRAIN)
    return {
        "pyproject.toml": f"""\
# This entry's dependencies. uv.lock pins every version, and our run installs
# exactly those. On Linux it installs the CUDA 13 build of JAX, which our run
# needs.
[project]
name = "{project}"
version = "0.1.0"
description = "A MARL-BattleGrounds leaderboard entry."
requires-python = ">=3.12"
dependencies = [
    "marl-battlegrounds[training,hydra,cuda13]=={version}; sys_platform == 'linux'",
    "marl-battlegrounds[training,hydra]=={version}; sys_platform != 'linux'",
]
""",
        ".python-version": "3.14\n",
        ".gitignore": """\
# Training output, the local environment and Python caches stay out of Git.
runs/
.venv/
__pycache__/
""",
        _ENTRY_FILE: f"""\
# Your leaderboard entry. `entry check` and our run read this file.

# Your method's name: your submission's title, your name on the leaderboard
# and the end of your Hugging Face repo's name. "My_MAPPO" becomes
# tdm-season-<N>-my-mappo, where <N> is the season's number.
name: {json.dumps(name)}

# The one command our run trains with. Change it to train your own method:
# `entry submit` refuses this starter's command. It must save its models in
# the output folder below.
train: {train}

# The folder the train command saves into, inside this repository. Commit
# nothing in it: our run needs it empty.
output: runs/entry

# The function that loads the model your run chose, written as
# module:function. It must finish within 10 minutes. This one is in
# my_entry/__init__.py.
load: my_entry:load

# Learned, Scripted, Hybrid or Language Model, in any letter case. The
# leaderboard shows it beside your entry.
entry_type: learned

# Every model your run needs from Hugging Face. Our run has no internet, so we
# download these before it starts: each released model it trains with or
# against, and each public outside model by its exact address. Your code must
# load an outside model at that same commit. Files in this repository need no
# line, and this starter needs none. To add some, write them like this:
#   starting_models:
#     - MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final
#     - MARL-BattleGrounds/tdm-season-0-results/training-opponents
#     - hf://<owner>/<repo>@<40-character commit>/<file, folder or *>
starting_models: []
""",
        "my_entry/__init__.py": '''\
"""Load this entry: the model its training run counts as its choice.

The train command in entry.yaml writes its run to runs/entry. While it trains,
training.current_choice names the best saved model so far; once it finishes,
the model the trainer chose.
"""

from pathlib import Path

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training

RUN = Path(__file__).resolve().parent.parent / "runs" / "entry"


def load() -> marl_bgs.System:
    """Return this entry's chosen model as a System.

    Raises
    ------
    RuntimeError
        The run has saved no choice yet. Its first one comes with its first
        validation, about 4 million steps in.
    """
    choice = training.current_choice(RUN)
    if choice is None:
        raise RuntimeError(f"No choice saved yet in {RUN}; train longer first.")
    return training.load_system(choice)
''',
        "README.md": f"""\
# {name}

A MARL-BattleGrounds leaderboard entry, made by `entry new`. It trains the
released Recurrent MAPPO recipe against itself, its past saves and the three
scripted bots.

## Enter It

Read the [rules](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/tournament_rules.md)
first. Then put your own method in: change the `train` command in
`entry.yaml`, and `my_entry/` if your run saves its models another way.
`entry submit` refuses the starter unchanged. A copy that changes only its
settings, such as the seed, is a duplicate, and we skip it.

Then, in this folder:

```sh
git add . && git commit -m "My method"
uv run python -m marl_battlegrounds entry check     # check it the way we will
git remote add origin <your GitHub repository URL>  # once
git push -u origin HEAD
uv run python -m marl_battlegrounds entry submit    # open the submission form
```

No training at home is needed. `entry check` installs your commit in a fresh
folder, trains it for 15 minutes, then loads and exports your model and
measures its play speed; our run trains it for up to
{OFFICIAL_TRAIN_MINUTES / 60:g} hours. To train at home anyway, run the
`train` command in `entry.yaml`.

## About This Starter

- It trains for 250,675,200 game steps, and its validation games add at most
  1,065,600 more, so it stays under the limit of 255 million.
- It exports a model at every save, every 409,600 steps
  (`retain_all_actors: true`). The rules ask for one at least every 5 million
  steps, so your method must save that often too.
- The output folder must be new or empty. For a second run at home, change
  `runs/entry` everywhere in `entry.yaml` and in `my_entry/__init__.py`.
""",
    }


def new_entry(path: str | os.PathLike[str], *, name: str | None = None) -> Path:
    """Write a starter entry in a new folder, make it a Git repository and lock it.

    The starter trains the released Recurrent MAPPO recipe against the
    scripted pool (itself, its past saves and the three scripted bots) and is
    meant to pass ``entry check`` on an RTX 5090. ``entry submit`` refuses it
    until its train command changes.

    Parameters
    ----------
    path : str or os.PathLike
        The folder to create. It must not exist yet, or be an empty folder.
    name : str or None, default=None
        The method name: the submission's title and the end of its Hugging
        Face repo name, ``tdm-season-<N>-<name>``, with the name in lowercase
        and each run of other characters made one hyphen. None uses the
        folder's name.

    Returns
    -------
    Path
        The entry folder, as an absolute path.

    Raises
    ------
    ValueError
        ``path`` is a file or a folder that is not empty, or the method name
        gives no repo name (it is blank, has no letters or digits, or gives
        more than 64 characters).
    RuntimeError
        Git is not installed, ``git init`` failed, or ``uv lock`` failed (its
        last lines are in the message; the written files stay).

    Notes
    -----
    Writes ``pyproject.toml`` (marl-battlegrounds at this installed version,
    with the CUDA 13 build on Linux), ``.python-version`` (3.14),
    ``.gitignore``, ``entry.yaml``, ``my_entry/__init__.py`` and
    ``README.md``. Then it runs ``git init`` and, when uv is on PATH,
    ``uv lock``, which downloads package metadata from PyPI. When uv is
    missing it prints the one-line uv install (``UV_INSTALL``) and
    ``UV_INSTALL_NOTE``, and stops before locking.
    It prints the method's repo name and the next commands. Nothing is
    committed. No GPU is used.

    Examples
    --------
    >>> from marl_battlegrounds import entry
    >>> entry.new_entry("my-mappo", name="My MAPPO")  # doctest: +SKIP
    """
    folder = Path(path).expanduser().absolute()
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise ValueError(f"{folder} must be a new or empty folder")
    chosen = (folder.name if name is None else name).strip()
    repo = _repo_name(chosen)
    if shutil.which("git") is None:
        raise RuntimeError("entry new needs Git. Install Git, then run it again.")
    project = re.sub(r"[^A-Za-z0-9._-]+", "-", folder.name).strip("-._") or "entry"
    for relative, content in _starter_files(
        chosen, project, _package_version()
    ).items():
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(folder, "init", "--quiet")
    print(f"Wrote a starter entry in {folder}")
    print(
        f"Your method {chosen!r} gets the Hugging Face repo "
        f"{_REPO_PATTERN.format(repo)}, where <N> is the season's number."
    )
    if shutil.which("uv") is None:
        print(f"uv is not installed. Install it with:\n  {UV_INSTALL}")
        print(UV_INSTALL_NOTE)
        print(f"Then lock the entry:\n  cd {folder}\n  uv lock")
        return folder
    done = subprocess.run(
        ["uv", "lock"], cwd=folder, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"uv lock failed:\n{_tail(done.stderr)}\n"
            f"Fix the cause, then run uv lock in {folder}."
        )
    print(
        "Next:\n"
        f"  cd {folder}\n"
        '  git add . && git commit -m "First entry"\n'
        "  uv run python -m marl_battlegrounds entry check"
    )
    return folder


def _machine() -> str:
    """Return the name of the first GPU this process may use, for the record.

    An empty or "-1" first entry in ``CUDA_VISIBLE_DEVICES`` hides every GPU,
    so the result is "No GPU visible". Otherwise ``nvidia-smi`` is asked for
    the first visible device (its index or UUID), or for the first GPU when
    the variable is not set. "No NVIDIA GPU found" when nvidia-smi is missing,
    fails or lists nothing. With mixed cards, CUDA numbers devices fastest
    first unless ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, and nvidia-smi numbers them
    by PCI bus, so the name can be another card's; it is only a record.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    command = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    if visible is not None:
        first = visible.split(",")[0].strip()
        if first in ("", "-1"):
            return "No GPU visible"
        command.append(f"--id={first}")
    if shutil.which("nvidia-smi") is not None:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            done = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=60
            )
            names = done.stdout.strip().splitlines()
            if done.returncode == 0 and names:
                return names[0].strip()
    return "No NVIDIA GPU found"


_FOREIGN_SETTINGS = (
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
    "PYTHONPATH",
    "PYTHONHOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
)


def _own_environment() -> dict[str, str]:
    """Return this process's environment without settings that leak into the entry.

    ``VIRTUAL_ENV``, ``UV_PROJECT_ENVIRONMENT``, ``PYTHONPATH`` and
    ``PYTHONHOME`` would run the entry's code in another Python environment;
    ``HF_HUB_CACHE`` and ``HUGGINGFACE_HUB_CACHE`` would let it read models
    from a shared cache instead of the check's fresh ``HF_HOME``, so an
    undeclared model could pass here and fail in the official run.
    """
    return {
        key: value for key, value in os.environ.items() if key not in _FOREIGN_SETTINGS
    }


def entry_environment(clone: Path, *, offline: bool) -> dict[str, str]:
    """Return the environment for a child that runs in a clone's own environment.

    ``check_entry`` gives it to every child that runs the entry's code, and a
    maintainers' script that runs more of the entry's code after a check (for
    example an export) must use it too, so both see the same model cache.

    Parameters
    ----------
    clone : Path
        The entry's clone. Its ``.venv`` comes first on ``PATH`` and becomes
        ``VIRTUAL_ENV``, and its ``.entry-check-hf`` folder is ``HF_HOME``, the
        model cache that step 4 fills.
    offline : bool
        True sets ``HF_HUB_OFFLINE=1`` and ``UV_OFFLINE=1``, as in steps 5 to 9
        and our run. False removes both, even when this process has them.

    Returns
    -------
    dict of str to str
        A new dict: this process's environment without the settings that would
        point the entry at another Python environment or model cache (see
        _own_environment), plus ``VIRTUAL_ENV``, ``PATH``, ``HF_HOME``,
        ``UV_NO_SYNC=1`` and the offline settings. Nothing is read from disk.
    """
    env = _own_environment()
    venv = clone / ".venv"
    env.update(
        VIRTUAL_ENV=str(venv),
        PATH=f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}",
        HF_HOME=str(clone / _HF_FOLDER),
        UV_NO_SYNC="1",
    )
    for key in ("HF_HUB_OFFLINE", "UV_OFFLINE"):
        if offline:
            env[key] = "1"
        else:
            env.pop(key, None)
    return env


def _released_package(lock: Mapping[str, Any]) -> str:
    """Return the marl-battlegrounds version that a parsed uv.lock installs.

    An entry trains and plays on a released marl-battlegrounds from PyPI,
    never on a changed copy. Every package named marl-battlegrounds in
    the lock must come from the PyPI index, and each of its files must be that
    version's own file on PyPI's file server
    (``https://files.pythonhosted.org/``), so ``uv sync --locked`` can only
    install the published package. Any released version passes, older or
    newer than the one running the check: whether an older release still
    suits a season is the maintainers' call, not this check's. Step 4 calls
    this before installing. It reads only ``lock``: no network and no files.

    Parameters
    ----------
    lock : Mapping
        The whole ``uv.lock`` file as ``tomllib`` parses it. Its
        ``[[package]]`` tables sit under the key ``"package"``.

    Returns
    -------
    str
        The locked version, for example ``"1.0.1"``. A lock that holds
        different versions for different platforms gives them all, sorted
        and joined by ", ".

    Raises
    ------
    _StepFailedError
        The lock installs no marl-battlegrounds; installs it from somewhere
        other than PyPI (a Git repository, a folder, a web address or another
        index); or names a file that is not on PyPI's file server or is not
        that version's own file.
    """
    packages = cast(list[dict[str, Any]], lock.get("package", []))
    found = [item for item in packages if item.get("name") == "marl-battlegrounds"]
    if not found:
        raise _StepFailedError(
            "uv.lock does not install marl-battlegrounds. Depend on it in "
            "pyproject.toml, then run uv lock."
        )
    versions: set[str] = set()
    for package in found:
        version = str(package.get("version", ""))
        source = cast(dict[str, object], package.get("source", {}))
        if source != {"registry": _PYPI_INDEX}:
            where = ", ".join(f"{kind} {value}" for kind, value in source.items())
            raise _StepFailedError(
                f"uv.lock installs marl-battlegrounds from {where or 'nowhere'}, "
                "not from PyPI. Depend on a released version, for example "
                f"marl-battlegrounds=={_package_version()}, then run uv lock."
            )
        stem = f"marl_battlegrounds-{version}"
        files = [package.get("sdist"), *package.get("wheels", [])]
        for url in (str(file.get("url", "")) for file in files if file):
            name = url.rsplit("/", 1)[-1]
            if not url.startswith(_PYPI_FILES) or not (
                name == f"{stem}.tar.gz" or name.startswith(f"{stem}-")
            ):
                raise _StepFailedError(
                    f"uv.lock fetches marl-battlegrounds {version} from {url}, "
                    "which is not its file on PyPI. Run uv lock again."
                )
        versions.add(version)
    return ", ".join(sorted(versions))


# Child programs run inside the entry's environment with ``python -c``.
_LOAD_CODE = """\
import json, sys
from marl_battlegrounds._method_loading import load_factory
from marl_battlegrounds.evaluation.policy_execution import Policy
method = load_factory(sys.argv[1])
print(json.dumps({"kind": "Policy" if isinstance(method, Policy) else "System"}))
"""
_EXPORT_CODE = """\
import json, sys
from marl_battlegrounds import export_model
from marl_battlegrounds._method_loading import load_factory
from marl_battlegrounds.evaluation.policy_execution import Policy
source, target = sys.argv[1], sys.argv[2]
if ":" in source and "/" not in source:
    source = load_factory(source)
    if not isinstance(source, Policy):
        print(json.dumps({"exported": False}))
        raise SystemExit(0)
export_model(source, target)
print(json.dumps({"exported": True}))
"""
_FETCH_CODE = """\
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from marl_battlegrounds.released_models import _parse_address
address = _parse_address(sys.argv[1])
kind = "dataset" if address.dataset else "model"
patterns = [address.path, address.path + "/*"]
root = Path(snapshot_download(
    address.repo, repo_type=kind, revision=address.commit,
    allow_patterns=patterns, token=False,
))
if address.path == "*":
    found = any(path.is_file() for path in root.rglob("*"))
else:
    found = (root / address.path).exists()
if not found:
    sys.exit(f"{sys.argv[1]} has no file or folder {address.path} at that commit.")
"""
_CHOICE_CODE = """\
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
builtin, choice = (run / "run_details.json").is_file(), None
if builtin:
    from marl_battlegrounds.training import current_choice
    choice = current_choice(run)
print(json.dumps({"builtin": builtin, "choice": choice and str(choice)}))
"""
_PLAY_CODE = """\
import json, sys
import jax
if jax.default_backend() != "gpu":
    print(json.dumps({"on_gpu": False, "device": jax.devices()[0].device_kind}))
    raise SystemExit(0)
from marl_battlegrounds.entry import play_limits
print(json.dumps(play_limits(sys.argv[1]).to_dict()))
"""


class _Checker:
    """Run the steps of one entry check and keep their results (see check_entry).

    One checker serves one call. ``work`` is the call's work folder (temporary,
    or the kept folder); the clone goes to ``work/entry`` and its Hugging Face
    cache to ``entry/.entry-check-hf``. ``bundle`` is where step 2 keeps a copy
    of the commit (maintainers only), or None. ``sandbox`` seals every child
    that runs the entry's code (see Sandbox), or None.
    """

    def __init__(
        self,
        source: str,
        commit: str | None,
        work: Path,
        train_minutes: float,
        bundle: Path | None,
        sandbox: Sandbox | None = None,
    ) -> None:
        self.source, self.commit, self.train_minutes = source, commit, train_minutes
        self.work, self.clone, self.bundle = work, work / "entry", bundle
        self.sandbox = sandbox
        self.hf_home = self.clone / _HF_FOLDER
        self.entry: _Entry | None = None
        self.stopped = False
        self.train_seconds = 0.0
        self.builtin_run = False
        self.choice: Path | None = None
        self.play: PlayLimits | None = None
        self.steps: dict[int, CheckStep] = {}

    def record(self, number: int, status: StepStatus, detail: str) -> None:
        """Save one step's result and print it as one line."""
        _, title, group = _STEPS[number - 1]
        self.steps[number] = CheckStep(number, title, group, status, detail)
        print(f"Step {number}. {title}: {status}. {detail}", flush=True)

    def attempt(self, number: int, step: Callable[[], str]) -> bool:
        """Run one must-pass step, record passed or failed, and say if it passed.

        A _StepFailedError message becomes the detail as written; any other error
        (from researcher code or the sandbox) is shown with its type, so it
        cannot crash the check.
        """
        try:
            detail = step()
        except _StepFailedError as error:
            self.record(number, "failed", str(error))
            return False
        except Exception as error:
            self.record(number, "failed", f"{type(error).__name__}: {error}")
            return False
        self.record(number, "passed", detail)
        return True

    def skip_rest(self) -> None:
        """Mark every step without a result as skipped, naming the first failure."""
        failed = [n for n, step in self.steps.items() if step.status == "failed"]
        reason = f"Not run: step {min(failed)} failed." if failed else "Not run."
        for number in range(1, len(_STEPS) + 1):
            if number not in self.steps:
                self.record(number, "skipped", reason)

    def sealed(self, command: Sequence[str], *, online: bool) -> list[str]:
        """Return ``command`` behind the sandbox's prefix (see Sandbox), if any."""
        prefix = () if self.sandbox is None else self.sandbox(self.clone, online)
        return [*prefix, *command]

    def python(
        self,
        code: str,
        *args: str,
        timeout: float,
        extra: Mapping[str, str] | None = None,
    ) -> tuple[int | None, dict[str, Any] | None, str]:
        """Run ``code`` offline and sealed with ``uv run --no-sync python -c``.

        ``args`` follow the code as ``sys.argv[1:]``; ``extra`` adds
        environment variables. Returns (exit code, or None after ``timeout``
        seconds; the JSON object on the last line of output, or None; the
        last lines of its error output).
        """
        command = ["uv", "run", "--no-sync", "python", "-c", code, *args]
        code_out, out, err = _run(
            self.sealed(command, online=False),
            cwd=self.clone,
            env={**entry_environment(self.clone, offline=True), **(extra or {})},
            timeout=timeout,
        )
        return code_out, _last_json(out) if code_out == 0 else None, _tail(err)

    def step_1(self, folder: Path) -> str:
        """Step 1 for a local repository: a clean tree and its last commit."""
        root = Path(_git(folder, "rev-parse", "--show-toplevel"))
        self.source = str(root)
        try:
            self.commit = _git(root, "rev-parse", "--verify", "HEAD")
        except _StepFailedError:
            raise _StepFailedError(
                "There is no commit yet. Commit your entry first."
            ) from None
        changes = _git(root, "status", "--porcelain").splitlines()
        if changes:
            raise _StepFailedError(
                "Commit or remove these changes first: "
                + "; ".join(line[3:] for line in changes[:5])
                + ("; ..." if len(changes) > 5 else "")
            )
        return f"Commit {self.commit}."

    def step_2(self) -> str:
        """Clone the commit into the work folder; keep a bundle if asked.

        A committed link that points outside the repository fails the step.
        """
        assert self.commit is not None
        _git(self.work, "clone", "--quiet", "--no-checkout", self.source, "entry")
        _git(self.clone, "checkout", "--quiet", "--detach", self.commit)
        if _git(self.clone, "rev-parse", "HEAD") != self.commit:
            raise _StepFailedError(f"The clone is not at {self.commit}.")
        # This process reads committed files (entry.yaml, uv.lock), so a link
        # in the commit must not lead it to a file outside the repository.
        top = self.clone.resolve()
        outside = [
            line.split("\t", 1)[1]
            for line in _git(self.clone, "ls-files", "-s", "-z").split("\0")
            if line.startswith("120000 ")
            and not (self.clone / line.split("\t", 1)[1]).resolve().is_relative_to(top)
        ]
        if outside:
            raise _StepFailedError(
                "The commit has links that point outside the repository: "
                + ", ".join(outside[:5])
            )
        if self.bundle is None:
            return f"Cloned {self.commit} into {self.clone}."
        self.bundle.parent.mkdir(parents=True, exist_ok=True)
        _git(self.clone, "bundle", "create", str(self.bundle), "HEAD")
        return f"Cloned {self.commit}; kept a copy in {self.bundle}."

    def step_3(self) -> str:
        """Read entry.yaml; the output folder must hold no committed files."""
        if not (self.clone / _ENTRY_FILE).is_file():
            raise _StepFailedError(f"No {_ENTRY_FILE} at the top of the repository.")
        try:
            self.entry = _read_entry(self.clone)
        except ValueError as error:
            raise _StepFailedError(str(error)) from None
        output = self.clone / self.entry.output
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise _StepFailedError(
                f"The commit already has files in the output folder "
                f"{self.entry.output}; the run needs it new or empty."
            )
        return (
            f"Method {self.entry.name!r} (Hugging Face repo "
            f"{_REPO_PATTERN.format(self.entry.repo)}); load {self.entry.load}."
        )

    def step_4(self) -> str:
        """Check the locked marl-battlegrounds, install, then fetch the models.

        Before installing, uv.lock must exist and install a released
        marl-battlegrounds from PyPI (``_released_package``). Then
        ``uv sync --locked`` runs with the GPU hidden, uv.lock must hold a CUDA
        build of JAX, and every starting model is downloaded (see fetch). The
        install and every download share one hour.
        """
        assert self.entry is not None
        if shutil.which("uv") is None:
            raise _StepFailedError(
                "uv is not installed. Install it with:\n"
                f"  {UV_INSTALL}\n{UV_INSTALL_NOTE}"
            )
        lock_file = self.clone / "uv.lock"
        if not lock_file.is_file():
            raise _StepFailedError(
                "No uv.lock at the top of the repository. Run uv lock, then "
                "commit uv.lock."
            )
        lock = tomllib.loads(lock_file.read_text(encoding="utf-8"))
        version = _released_package(lock)
        env = _own_environment()
        env["CUDA_VISIBLE_DEVICES"] = ""
        started = time.monotonic()
        deadline = started + _INSTALL_SECONDS
        code, _, err = _run(
            self.sealed(["uv", "sync", "--locked"], online=True),
            cwd=self.clone,
            env=env,
            timeout=_INSTALL_SECONDS,
        )
        seconds = time.monotonic() - started
        if code is None:
            raise _StepFailedError(_INSTALL_LIMIT)
        if code != 0:
            raise _StepFailedError(f"uv sync --locked failed:\n{_tail(err)}")
        # uv sync --locked never rewrites uv.lock, so the parse above still holds.
        packages = cast(list[dict[str, object]], lock.get("package", []))
        if not {str(item.get("name")) for item in packages} & _CUDA_JAX:
            raise _StepFailedError(
                "uv.lock installs no CUDA build of JAX. On Linux, depend on "
                "marl-battlegrounds[cuda13] (or cuda12), then run uv lock."
            )
        self.hf_home.mkdir(parents=True, exist_ok=True)
        if self.hf_home.is_symlink():
            # The install ran entry code; our own downloads must not follow it out.
            raise _StepFailedError(f"The install made {_HF_FOLDER} a link.")
        for model in self.entry.starting_models:
            self.fetch(model, deadline)
        return (
            f"marl-battlegrounds {version} from PyPI. Installed in "
            f"{seconds / 60:.1f} minutes without the GPU; "
            f"{len(self.entry.starting_models)} starting models downloaded."
        )

    def fetch(self, model: str, deadline: float) -> None:
        """Download one starting model into the check's HF_HOME (step 4).

        A MARL-BattleGrounds name or address goes through the entry's own
        ``models download``, sealed with the network on, then once more
        offline: a name the entry's package does not fix to one commit cannot
        load in our offline run, so step 4 fails it now with released_models'
        own advice. Any other owner's ``hf://`` address is fetched by the
        check's own Python in a child (no entry code runs): one
        ``snapshot_download`` of the path (a file, a folder, or ``*`` for the
        whole repo) at its commit, with no token, so only public files pass,
        and a path with no files fails. Every child stops when step 4's hour
        ends.
        """
        address = _parse_address(model) if model.startswith("hf://") else None
        if address is not None:
            owner = address.repo.split("/")[0]
            if owner.lower() != RELEASE_ORGANIZATION.lower():
                self.child(
                    [sys.executable, "-I", "-c", _FETCH_CODE, model],
                    f"Downloading {model}",
                    deadline,
                    cwd=self.work,
                    env=entry_environment(self.clone, offline=False),
                )
                return
        command = ["uv", "run", "--no-sync", "python", "-m", "marl_battlegrounds"]
        download = [*command, "models", "download", model]
        for offline in (False, True):
            self.child(
                self.sealed(download, online=not offline),
                f"models download {model}" + (" offline" if offline else ""),
                deadline,
                cwd=self.clone,
                env=entry_environment(self.clone, offline=offline),
            )

    def child(
        self,
        command: Sequence[str],
        what: str,
        deadline: float,
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> None:
        """Run one step 4 child until ``deadline`` (a ``time.monotonic`` value).

        Raises _StepFailedError with the hour's message when the time runs
        out, or with ``what`` and the last lines of its error output when it
        fails.
        """
        timeout = max(deadline - time.monotonic(), 0.0)
        code, _, err = _run(command, cwd=cwd, env=env, timeout=timeout)
        if code is None:
            raise _StepFailedError(_INSTALL_LIMIT)
        if code != 0:
            raise _StepFailedError(f"{what} failed:\n{_tail(err)}")

    def step_5(self) -> str:
        """Train offline for train_minutes, then stop the run's process group."""
        assert self.entry is not None
        log_path = self.work / "train.log"
        calm = True
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                self.sealed(["/bin/sh", "-c", self.entry.train], online=False),
                cwd=self.clone,
                env=entry_environment(self.clone, offline=True),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with _group_stopped_on_error(process):
                try:
                    code = process.wait(timeout=self.train_minutes * 60.0)
                except subprocess.TimeoutExpired:
                    code = None
                    calm = _stop_group(process, _STOP_GRACE_SECONDS)
                else:
                    # Programs the train command left behind in its group stop too.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
        self.train_seconds = time.monotonic() - started
        self.stopped = code is None
        if code is not None and code != 0:
            output = _tail(log_path.read_text(encoding="utf-8", errors="replace"))
            raise _StepFailedError(
                f"The train command failed after {self.train_seconds / 60:.1f} "
                f"minutes (exit code {code}):\n{output}"
            )
        if code == 0:
            return (
                f"The train command ended by itself after "
                f"{self.train_seconds / 60:.1f} minutes."
            )
        return (
            f"Trained for {self.train_minutes:g} minutes with downloads off "
            "(our run is offline), then "
            + (
                "stopped within 60 seconds of the stop signal."
                if calm
                else "did not stop within 60 seconds of the stop signal and was killed."
            )
        )

    def find_choice(self) -> None:
        """Ask the trainer's own ranking for a built-in run's choice (before step 6).

        A built-in run is an output folder with ``run_details.json``; its
        choice is what ``training.current_choice`` names, or None. The entry's
        run wrote those files, so a sealed child reads them, never this
        process.

        Raises
        ------
        RuntimeError
            The child failed or ran past 10 minutes; the message has its last
            lines of error output.
        """
        assert self.entry is not None
        run = str(self.clone / self.entry.output)
        _, result, err = self.python(_CHOICE_CODE, run, timeout=_LOAD_SECONDS)
        if result is None:
            raise RuntimeError(
                err or f"no answer within {_LOAD_SECONDS / 60:g} minutes"
            )
        self.builtin_run = bool(result["builtin"])
        self.choice = None if result["choice"] is None else Path(result["choice"])

    def saved(self) -> str | None:
        """Step 8's detail when the run saved something to enter, else None.

        A built-in run must have a choice; an own training loop must have
        written something into its output folder.
        """
        assert self.entry is not None
        if self.builtin_run:
            return (
                None
                if self.choice is None
                else f"The run's choice is {self.choice.name}."
            )
        output = self.clone / self.entry.output
        if output.is_dir() and any(output.iterdir()):
            return f"The output folder {self.entry.output} holds the run's saves."
        return None

    def step_6(self) -> str:
        """Call the load function in a fresh process, within 10 minutes."""
        assert self.entry is not None
        code, result, err = self.python(
            _LOAD_CODE, self.entry.load, timeout=_LOAD_SECONDS
        )
        if code is None:
            raise _StepFailedError("Loading took more than 10 minutes.")
        if result is None:
            raise _StepFailedError(f"{self.entry.load} failed:\n{err}")
        return f"{self.entry.load} returned a {result['kind']}."

    def step_7(self) -> str:
        """Export the choice (or the loaded Policy); refuse local paths inside it.

        The sealed export child writes inside the clone, the only folder it
        may write. This process copies the file to ``work/model.zip``, which
        step 9 plays, and refuses a link or anything else that is not a plain
        file, so entry code cannot make it copy one of our own files.
        """
        assert self.entry is not None
        target = self.clone / _EXPORT_FILE
        source = str(self.choice) if self.choice is not None else self.entry.load
        code, result, err = self.python(
            _EXPORT_CODE, source, str(target), timeout=_EXPORT_SECONDS
        )
        if code is None:
            raise _StepFailedError("Exporting took more than 10 minutes.")
        if result is None:
            raise _StepFailedError(f"Exporting failed:\n{err}")
        if not result["exported"]:
            raise _StepFailedError(
                f"{self.entry.load} returned a team System: return a per-agent "
                "Policy so we can export it."
            )
        from marl_battlegrounds.released_models import (
            _local_path_findings,  # pyright: ignore[reportPrivateUsage]
        )

        kept = self.work / "model.zip"
        # Entry code may have swapped the export for a link to one of our files.
        try:
            handle = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            raise _StepFailedError(
                f"The export did not leave a plain file at {target}."
            ) from None
        with os.fdopen(handle, "rb") as exported:
            if not stat.S_ISREG(os.fstat(exported.fileno()).st_mode):
                raise _StepFailedError(
                    f"The export did not leave a plain file at {target}."
                )
            with kept.open("wb") as copy:
                shutil.copyfileobj(exported, copy)
        findings: list[str] = _local_path_findings(kept.name, kept.read_bytes())
        if findings:
            raise _StepFailedError(
                " ".join(
                    f"{kept.name} names {finding}. Train against a released "
                    "name or a model inside your repository, or lock a newer "
                    "marl-battlegrounds, whose export leaves local folders out."
                    for finding in findings
                )
            )
        return (
            f"Exported {kept.stat().st_size:,} bytes from {Path(source).name} "
            f"to {kept}."
        )

    def step_9(self) -> tuple[StepStatus, str]:
        """Time the exported model (the file our games play) with deterministic XLA.

        A fresh child of this process's own Python plays the check's copy,
        ``work/model.zip``, so no entry code runs and the reading cannot be
        forged by the entry.

        Only a reading on an RTX 5090, the card our run uses, with its memory
        known, passes or fails; any other reading is "not measured". A crash
        or games past 30 minutes fail on any machine.
        """
        assert self.entry is not None
        flags = f"{os.environ.get('XLA_FLAGS', '')} {_DETERMINISTIC}".strip()
        env = {**_own_environment(), "XLA_FLAGS": flags, "HF_HUB_OFFLINE": "1"}
        export = self.work / "model.zip"
        code, out, err = _run(
            [sys.executable, "-I", "-c", _PLAY_CODE, str(export)],
            cwd=self.work,
            env=env,
            timeout=_PLAY_SECONDS,
        )
        result = _last_json(out) if code == 0 else None
        if code is None:
            return "failed", "The games took more than 30 minutes: far too slow."
        if result is None:
            return "failed", f"The play check failed:\n{_tail(err)}"
        if not result["on_gpu"]:
            return "not measured", (
                f"No GPU here (JAX found {result['device']}). We measure this on "
                "our RTX 5090."
            )
        self.play = PlayLimits(
            **{key: result[key] for key in PlayLimits.__dataclass_fields__}
        )
        speed, peak = self.play.game_steps_per_second, self.play.peak_memory_bytes
        detail = (
            f"On {self.play.device}: {speed:,.0f} game steps per second (at least "
            f"{MIN_GAME_STEPS_PER_SECOND:,} needed); peak "
            + ("unknown" if peak is None else f"{peak / 1e9:.2f} GB")
            + " of GPU memory (at most 10 GB allowed)."
        )
        if "RTX 5090" not in self.play.device:
            return "not measured", detail + " We decide this on our RTX 5090."
        if peak is None:
            return "not measured", detail + " The GPU did not report its memory."
        return ("passed" if self.play.within_limits else "failed"), detail

    def after_training(self) -> None:
        """Run steps 6 to 9 once step 5 passed.

        When the run saved nothing to enter (see saved), steps 6, 7 and 9
        wait (are skipped, not failed) if step 5 stopped it at its time limit,
        and step 8 says so; a run that ended by itself with nothing saved
        fails step 6. A run whose records or output folder cannot be read
        fails step 6. An error in step 9 itself (for example from the
        sandbox) fails step 9.
        """
        assert self.entry is not None
        try:
            self.find_choice()
            saved = self.saved()
        except Exception as error:
            self.record(6, "failed", f"The run's records could not be read: {error}")
            self.skip_rest()
            return
        if saved is None and not self.stopped:
            self.record(
                6,
                "failed",
                f"The train command ended after {self.train_seconds / 60:.1f} "
                f"minutes with nothing saved to enter in {self.entry.output}.",
            )
            self.skip_rest()
            return
        if saved is None:
            waiting = "Not checked: nothing was saved to enter yet (see step 8)."
            later = (
                f" Our run trains for up to {OFFICIAL_TRAIN_MINUTES / 60:g} hours."
                if self.train_minutes < OFFICIAL_TRAIN_MINUTES
                else ""
            )
            self.record(6, "skipped", waiting)
            self.record(7, "skipped", waiting)
            self.record(
                8,
                "not measured",
                f"Nothing saved to enter in {self.train_minutes:g} minutes of "
                f"training.{later}",
            )
            self.record(9, "skipped", waiting)
            return
        ok = self.attempt(6, self.step_6) and self.attempt(7, self.step_7)
        self.record(8, "passed", saved)
        if ok:
            try:
                self.record(9, *self.step_9())
            except Exception as error:
                self.record(9, "failed", f"{type(error).__name__}: {error}")
        self.skip_rest()


def check_entry(
    path: str | os.PathLike[str] = ".",
    *,
    repository: str | None = None,
    commit: str | None = None,
    train_minutes: float = 15.0,
    sandbox: Sandbox | None = None,
    keep_folder: str | os.PathLike[str] | None = None,
) -> EntryCheck:
    """Check one commit of an entry the way we will, and save the result.

    Steps 1 to 7 must pass on any machine; they block ``entry submit``.
    Steps 8 and 9 are measured here and decided on our RTX 5090; they never
    block. Each step prints one line as it finishes.

    1. The tree is clean and the commit is recorded.
    2. A fresh clone of that commit, with no committed link that points
       outside the repository.
    3. ``entry.yaml`` is present and valid: the six keys ``name``,
       ``train``, ``output``, ``load``, ``entry_type`` and
       ``starting_models`` (1.0.3's ``training_maps`` and ``config`` are
       ignored), and the output folder holds no committed files.
    4. ``uv.lock`` installs a released marl-battlegrounds from PyPI, any
       version (checked before installing); ``uv sync --locked`` with the
       GPU hidden; ``uv.lock`` holds a CUDA build of JAX; every starting
       model is downloaded into a fresh ``HF_HOME`` in the clone. A released
       name must also load offline: one the locked package does not fix to a
       commit fails here with the advice to list its exact ``hf://``
       address, in ``starting_models`` and wherever the run names it. An
       outside path with no files fails. The install and the downloads share
       1 hour.
    5. The train command runs for ``train_minutes`` with ``HF_HUB_OFFLINE=1``,
       then gets a stop signal (SIGINT) to its process group, 60 seconds'
       grace, then a kill.
    6. The load function, in a fresh process, returns a Policy or System
       within 10 minutes.
    7. Export to a portable model file: from the actor folder
       ``training.current_choice`` names when the output is a built-in run
       (read by a sealed child), else from the Policy that the load function
       returns. The file must be a plain file, not a link, and must name no
       folder on this computer.
    8. Something to enter was saved: a built-in run's choice (never its
       step-0 save), or files in an own training loop's output folder. When
       step 5 stopped the run before it saved anything (a slower card may
       not reach the first save), steps 6, 7 and 9 wait and step 8 is "not
       measured"; a run that ended by itself with nothing saved fails step 6.
    9. ``play_limits`` on the check's copy of the exported file, the one our
       games play, in a fresh child of this Python (no entry code runs), with
       XLA's deterministic mode.
       Its reading passes or fails only on an RTX 5090 that reports its
       memory; on another GPU it is "not measured" with the reading, and
       without a GPU "not measured". A crash, or games that run past 30
       minutes, fail it on any machine.

    Parameters
    ----------
    path : str or os.PathLike, default="."
        Without ``repository``: a folder inside the entry's Git repository;
        its last commit is checked. With ``repository``: the folder that gets
        the record and the kept copy of the commit.
    repository : str or None, default=None
        For maintainers: the URL (or path) of a repository to clone. Give it
        together with ``commit``. Step 1 then checks only the commit's form.
    commit : str or None, default=None
        The full 40-character commit SHA to check, with ``repository``.
    train_minutes : float, default=15.0
        How long step 5 trains, in minutes: 15 for the check,
        OFFICIAL_TRAIN_MINUTES (480) for our official run. Must be above 0.
    sandbox : callable or None, default=None
        For maintainers: a function ``sandbox(clone, online)`` that returns a
        command prefix (a list of strings, for example a ``bwrap`` command).
        The check puts it in front of every child that runs the entry's code
        (see Sandbox). ``clone`` is the clone's folder, the only folder those
        children write. ``online`` is True for step 4's install and
        released-model downloads and False for every child in steps 5 to 7.
        The prefix must keep the environment the check gives each child
        (``HF_HOME``, ``HF_HUB_OFFLINE`` and the rest) and keep the child in
        the check's process group. An error it raises fails that step. None
        runs the children unsealed.
    keep_folder : str or os.PathLike or None, default=None
        For maintainers: a new folder to work in and keep, for our official
        run. It holds the clone (``entry``, with its environment, the run's
        output and the Hugging Face cache), ``train.log`` and the exported
        ``model.zip``. Nothing in it is removed, even after an interrupt.
        None works in a temporary folder that is removed at the end.

    Returns
    -------
    EntryCheck
        Every step's result. ``passed`` is True when no must-pass step failed.

    Raises
    ------
    ValueError
        Only one of ``repository`` and ``commit`` is given, the commit is not
        a full SHA, ``train_minutes`` is not above 0, or ``keep_folder``
        already exists.
    RuntimeError
        Git is not installed. Problems inside the steps never raise: they
        become failed steps.
    OSError
        The record could not be written, or ``keep_folder`` could not be made.
    KeyboardInterrupt
        Ctrl-C, SIGTERM or SIGHUP stopped the check. The running child's whole
        process group (the training run in step 5) is killed first, and the
        temporary folder is removed (a kept folder stays). The CLI prints
        "Interrupted" and exits 130.

    Notes
    -----
    Signals: while the check runs in the main thread, SIGTERM and SIGHUP raise
    KeyboardInterrupt, as Ctrl-C does; the previous handlers come back
    afterwards. Called from another thread, Python's defaults stay.

    Files: the record is saved as JSON at
    ``.git/marl-bgs-entry-check/<commit>.json`` of the checked repository
    (never committed). With ``repository``, it goes to
    ``<path>/marl-bgs-entry-check/<commit>.json`` beside
    ``<commit>.bundle``, a ``git bundle`` of the commit, kept so the code we
    ran can be released. Nothing is saved when step 1 fails. Without
    ``keep_folder``, the clone, its environment (several GB with CUDA) and
    the run live in a temporary folder (set ``TMPDIR`` to choose where).

    Processes and network: steps 4 to 7 run the entry's own code in child
    processes inside the clone's locked environment (``uv run --no-sync``).
    Only steps 2 and 4 use the network: step 2 clones, step 4 installs and
    downloads. An outside ``hf://`` model (any owner but MARL-BattleGrounds) is
    downloaded by a child of this Python, with no token and outside any
    ``sandbox``; no entry code runs there. Step 9 also runs in a child of this
    Python, on the check's own copy of the export, so the entry's code cannot
    report its own speed. Steps 5 to 9 set ``HF_HUB_OFFLINE=1`` (steps 5 to 7
    also ``UV_OFFLINE=1``); this asks tools to stay offline but does not cut
    the network unless a ``sandbox`` does. Steps 5 to 9 may use the GPU.

    Examples
    --------
    >>> from marl_battlegrounds import entry
    >>> result = entry.check_entry("my-entry")  # doctest: +SKIP
    >>> result.passed  # doctest: +SKIP
    True
    """
    if (repository is None) != (commit is None):
        raise ValueError("Give both repository and commit, or neither")
    if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be the full 40-character SHA, in lowercase")
    if not train_minutes > 0:
        raise ValueError("train_minutes must be above 0")
    kept = None if keep_folder is None else Path(keep_folder).expanduser().absolute()
    if kept is not None and kept.exists():
        raise ValueError(f"keep_folder {kept} already exists; give a new folder")
    if shutil.which("git") is None:
        raise RuntimeError("entry check needs Git. Install Git, then run it again.")
    folder = Path(path).expanduser().absolute()
    machine = _machine()
    records = None if repository is None else folder / _RECORD_FOLDER
    work_folder: contextlib.AbstractContextManager[str]
    if kept is None:
        work_folder = tempfile.TemporaryDirectory(
            prefix="marl-bgs-entry-check-", ignore_cleanup_errors=True
        )
    else:
        kept.mkdir(parents=True)
        work_folder = contextlib.nullcontext(str(kept))
    with _stop_signals_interrupt(), work_folder as work:
        checker = _Checker(
            repository or str(folder),
            commit,
            Path(work),
            train_minutes,
            None if records is None else records / f"{commit}.bundle",
            sandbox,
        )
        if repository is None:
            ok = checker.attempt(1, lambda: checker.step_1(folder))
            if ok:
                records = _record_folder(Path(checker.source))
        else:
            checker.record(1, "passed", f"Checking {repository} at {commit}.")
            ok = True
        for number, step in (
            (2, checker.step_2),
            (3, checker.step_3),
            (4, checker.step_4),
            (5, checker.step_5),
        ):
            ok = ok and checker.attempt(number, step)
        if ok:
            checker.after_training()
        checker.skip_rest()
    result = EntryCheck(
        repository=repository or checker.source,
        commit=checker.commit,
        machine=machine,
        checked_at=datetime.now(UTC).isoformat(timespec="seconds"),
        package_version=_package_version(),
        steps=tuple(checker.steps[number] for number in sorted(checker.steps)),
        play=checker.play,
    )
    if records is not None and checker.commit is not None:
        records.mkdir(parents=True, exist_ok=True)
        target = records / f"{checker.commit}.json"
        result = replace(result, record=str(target))
        text = json.dumps(result.to_dict(), indent=2) + "\n"
        target.write_text(text, encoding="utf-8")
    verdict = (
        "Passed: no must-pass step failed."
        if result.passed
        else "Failed: a must-pass step failed."
    )
    waiting = [step.number for step in result.must_pass if step.status == "skipped"]
    if result.passed and waiting:
        verdict += (
            f" Steps {', '.join(map(str, waiting))} were not checked: nothing was "
            "saved to enter yet (see step 8)."
        )
    print(verdict + (f" Record: {result.record}" if result.record else ""))
    return result


def _record_folder(root: Path) -> Path:
    """Return ``<the repository's .git folder>/marl-bgs-entry-check``.

    ``root`` is the repository's top folder. Worktrees share their main
    repository's folder. Nothing is created.
    """
    common = _git(root, "rev-parse", "--git-common-dir")
    return (root / common).resolve() / _RECORD_FOLDER


def _remote_url(root: Path, commit: str) -> str:
    """Return the web address of the remote whose branch holds ``commit``.

    Raises ValueError when no remote branch holds the commit. An SSH address
    such as ``git@github.com:you/entry.git`` becomes
    ``https://github.com/you/entry``. A user name or token inside the
    address (``https://user:TOKEN@github.com/...``) is removed, so it never
    reaches the printed link or the public issue.
    """
    branches = _git(root, "branch", "-r", "--contains", commit).splitlines()
    branches = [line.strip() for line in branches if "->" not in line]
    if not branches:
        raise ValueError(
            "This commit is on no remote branch. Push it first: git push -u "
            "origin HEAD (add the remote once with git remote add origin "
            "<your GitHub repository URL>)."
        )
    url = _git(root, "remote", "get-url", branches[0].split("/", 1)[0])
    ssh = re.fullmatch(r"(?:ssh://)?git@([^:/]+)[:/](.+)", url)
    if ssh:
        url = f"https://{ssh.group(1)}/{ssh.group(2)}"
    parts = urllib.parse.urlsplit(url)
    if parts.scheme in ("http", "https") and "@" in parts.netloc:
        host = parts.hostname or ""
        url = parts._replace(
            netloc=host + (f":{parts.port}" if parts.port else "")
        ).geturl()
    return url.removesuffix(".git")


def _issue_link(values: Mapping[str, str]) -> str:
    """Return the issue-form link with ``values`` filled in and URL-encoded."""
    query = urllib.parse.urlencode(
        {"template": FORM_TEMPLATE, **values}, quote_via=urllib.parse.quote
    )
    return f"{FORM_URL}?{query}"


def submit_entry(
    path: str | os.PathLike[str] = ".", *, open_browser: bool = True
) -> str:
    """Open the leaderboard submission form with this entry's details filled in.

    Parameters
    ----------
    path : str or os.PathLike, default="."
        A folder inside the entry's Git repository. Its last commit is
        submitted.
    open_browser : bool, default=True
        Open the link in the default web browser (from WSL2, the Windows
        default browser). The link is printed either way.

    Returns
    -------
    str
        The link that was printed (and opened).

    Raises
    ------
    ValueError
        There is no commit yet; the tree has uncommitted changes;
        ``entry.yaml`` is not valid; the train command is still the starter's
        (it trains Season 0's Recurrent MAPPO recipe unchanged); the commit is
        on no remote branch (push it first); or this commit has no passing
        ``entry check`` record.
    FileNotFoundError
        There is no ``entry.yaml`` at the top of the repository.
    RuntimeError
        Git is not installed, or a Git command failed (for example, the
        folder is not inside a Git repository).

    Notes
    -----
    Fills the title (the method name) and the form's two text fields,
    ``repository`` and ``commit``; the maintainers read everything else from
    ``entry.yaml`` at that commit. On the page you tick the license box,
    and "My repository is private" if it is. Nothing is sent until you press
    the button on GitHub; the issue's number and time are the public arrival
    receipt. Reads only local Git data: it does not fetch from the remote.

    Examples
    --------
    >>> from marl_battlegrounds import entry
    >>> link = entry.submit_entry("my-entry")  # doctest: +SKIP
    """
    if shutil.which("git") is None:
        raise RuntimeError("entry submit needs Git. Install Git, then run it again.")
    root = Path(
        _git(Path(path).expanduser().absolute(), "rev-parse", "--show-toplevel")
    )
    try:
        commit = _git(root, "rev-parse", "--verify", "HEAD")
    except _StepFailedError:
        raise ValueError("There is no commit yet. Commit your entry first.") from None
    if _git(root, "status", "--porcelain"):
        raise ValueError("Commit your changes first: the tree has uncommitted changes.")
    entry = _read_entry(root)
    if entry.train == _STARTER_TRAIN:
        raise ValueError(
            "This is the starter: it trains Season 0's Recurrent MAPPO recipe "
            "unchanged. Put your own method into the train command in "
            "entry.yaml, then commit, check and push it before you submit."
        )
    repository = _remote_url(root, commit)
    record = _record_folder(root) / f"{commit}.json"
    passed = record.is_file() and json.loads(record.read_text(encoding="utf-8")).get(
        "passed"
    )
    if passed is not True:
        raise ValueError(
            "This commit has no passing check. Run "
            "`uv run python -m marl_battlegrounds entry check` first."
        )
    link = _issue_link(
        {"title": entry.name, "repository": repository, "commit": commit}
    )
    print(f"\nSubmission form for {entry.name} at {commit}:\n{link}")
    print(
        '\nOn the page, tick the license box, and "My repository is private" if '
        "it is. Then press Submit new issue."
    )
    if open_browser:
        open_in_browser(link)
    return link


def play_limits(reference: str, *, games: int = 512, batches: int = 4) -> PlayLimits:
    """Measure the play limits: a system plays itself; time it, read its peak memory.

    The system plays itself on maps 42-46 with paired spawns and priority
    metrics. A first call of ``games`` games compiles and warms up; then one
    timed call plays ``games * batches`` games, ``games`` at once.

    Parameters
    ----------
    reference : str
        Any reference ``evaluate`` takes: a built-in name, a saved actor
        folder, a portable model file, a released name or a ``module:function``
        load function. It is loaded once and plays both teams.
    games : int, default=512
        Games played at once, and the size of the warm-up batch. A positive
        even number (paired spawns come in pairs).
    batches : int, default=4
        How many batches of ``games`` the timed call plays. Positive.

    Returns
    -------
    PlayLimits
        The device, games, game steps, seconds and peak memory.
        ``within_limits`` says whether both limits hold on a GPU.

    Raises
    ------
    ValueError
        ``games`` is not a positive even number, ``batches`` is not positive,
        or JAX runs on a GPU without XLA's deterministic mode declared (set
        ``XLA_FLAGS=--xla_gpu_deterministic_ops=true`` before Python starts).
    TypeError
        The reference loads as something other than a Policy or System.

    Notes
    -----
    Runs on JAX's default device; on the CPU it still plays every game, but
    ``on_gpu`` is False, so it does not measure the limits. The peak memory is
    the whole process's peak so far, so call this in a fresh process (as
    ``entry check`` does) for a reading of this system alone. Seeds are fixed
    (1 for the warm-up, 2 for the timed call). Loading may download a
    released model; no files are written.

    Examples
    --------
    >>> from marl_battlegrounds import entry
    >>> entry.play_limits("tdm-alpha").game_steps_per_second  # doctest: +SKIP
    """
    import jax

    from marl_battlegrounds._method_loading import load_method
    from marl_battlegrounds.evaluation.evaluate import evaluate
    from marl_battlegrounds.evaluation.runtime_provenance import (
        declared_gpu_deterministic_ops,
        read_declared_xla_flags,
    )

    if isinstance(games, bool) or games <= 0 or games % 2:
        raise ValueError("games must be a positive even number")
    if isinstance(batches, bool) or batches <= 0:
        raise ValueError("batches must be positive")
    on_gpu = jax.default_backend() == "gpu"
    if on_gpu and not declared_gpu_deterministic_ops(read_declared_xla_flags()):
        raise ValueError(
            "set XLA_FLAGS=--xla_gpu_deterministic_ops=true before Python starts"
        )
    system = load_method(reference)

    def play(count: int, seed: int) -> tuple[float, int]:
        """Play ``count`` self-play games; return (seconds, game steps)."""
        started = time.perf_counter()
        result = evaluate(
            system,
            system,
            num_episodes=count,
            maps=PLAY_MAPS,
            spawn_mode="paired",
            seed=seed,
            num_envs=games,
            metrics="priority",
        )
        seconds = time.perf_counter() - started
        return seconds, sum(episode.episode_length for episode in result.episodes)

    warmup_seconds, _ = play(games, 1)
    seconds, steps = play(games * batches, 2)
    device = cast(Any, jax.local_devices()[0])
    stats = cast(dict[str, int] | None, device.memory_stats()) or {}
    peak = stats.get("peak_bytes_in_use")
    return PlayLimits(
        reference=reference,
        device=str(device.device_kind),
        on_gpu=on_gpu,
        games=games * batches,
        game_steps=int(steps),
        seconds=seconds,
        warmup_seconds=warmup_seconds,
        peak_memory_bytes=None if peak is None else int(peak),
    )


__all__ = [
    "CheckStep",
    "EntryCheck",
    "PlayLimits",
    "check_entry",
    "new_entry",
    "play_limits",
    "submit_entry",
]
