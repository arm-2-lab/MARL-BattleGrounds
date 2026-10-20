"""Enter the leaderboard: start an entry, check it the way we will, and submit it.

This module owns the three entry commands. The command line only parses and
calls these functions (``python -m marl_battlegrounds entry new|check|submit``):

- ``new_entry`` writes a starter repository that trains the released Recurrent
  MAPPO recipe, runs ``git init`` and locks it with ``uv lock``.
- ``check_entry`` runs the public check of the tournament rulebook on one
  commit, in nine numbered steps. Steps 1 to 7 must pass on any machine and
  block ``entry submit``. Steps 8 and 9 are measured here but decided on our
  RTX 5090, so they never block. The result is saved under the repository's
  ``.git`` folder, so it is never committed.
- ``submit_entry`` refuses unfinished work, then opens the GitHub issue form
  with the entry's details filled in.
- ``play_limits`` measures Rule 4: the system plays itself and we time it.

Who owns what. The rules live in ``docs/tournament_rules.md``; the rule numbers
here are that file's numbers. Existing owners do the real work: the entry's own
train command trains, ``load_factory`` (``module:function`` loading) loads,
``export_model`` exports, ``models download`` fetches released models and
training-opponent pools (``released_models`` decides which name is a pool),
``training.current_choice`` names a built-in run's choice, and ``evaluate``
plays the games. Researcher code (train, load, export, play) always runs in
child processes inside the entry's own locked environment, never in the
caller's Python. This module is POSIX only (it stops process groups), like the
rest of the package's run files.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import tomllib
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, cast

from marl_battlegrounds._browser import open_in_browser
from marl_battlegrounds.released_models import (
    _pool,  # pyright: ignore[reportPrivateUsage]
)

FORM_URL = "https://github.com/arm-2-lab/MARL-BattleGrounds/issues/new"
"""The GitHub page that opens a new issue in the MARL-BattleGrounds repository."""

FORM_TEMPLATE = "leaderboard_submission.yml"
"""The issue form's file name in ``.github/ISSUE_TEMPLATE``."""

FORM_FIELDS = (
    "repository",
    "commit",
    "train-command",
    "lock-file",
    "load-function",
    "config-and-maps",
    "starting-models",
)
"""The form's text fields ``submit_entry`` fills, by their IDs in the form file.

The form's other text field, ``team``, and its dropdown and checkboxes cannot
be filled by the team's computer: the team fills them in on the page.
"""

MAX_LINK_LENGTH = 8000
"""Longest link ``submit_entry`` opens, in characters. GitHub refuses longer ones."""

MIN_GAME_STEPS_PER_SECOND = 5000
"""Rule 4.1: the least playing speed, in game steps per second."""

MAX_GPU_MEMORY_BYTES = 10 * 10**9
"""Rule 4.2: the most GPU memory, in bytes (10 GB, with 1 GB = 10**9 bytes)."""

PLAY_MAPS = (42, 43, 44, 45, 46)
"""The validation maps ``play_limits`` plays on."""

ENTRY_TYPES = ("learned", "scripted", "hybrid", "language model")
"""The entry types of Rule 1.2, as written in ``entry.yaml`` (any letter case)."""

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
_SETTINGS = (
    "name",
    "train",
    "output",
    "load",
    "entry_type",
    "training_maps",
    "config",
    "starting_models",
)

StepGroup = Literal["must pass", "measured here"]
"""Must-pass steps block ``entry submit``; measured-here steps never do."""

StepStatus = Literal["passed", "failed", "skipped", "not measured"]
"""What happened to one step. "skipped" means it was not run; its detail says why."""

_STEPS: tuple[tuple[int, str, str, StepGroup], ...] = (
    (1, "6.3", "The tree is clean and the commit is recorded", "must pass"),
    (2, "6.3", "A fresh clone of the commit", "must pass"),
    (3, "3.1, 3.4, 6.3", "entry.yaml is present and valid", "must pass"),
    (4, "2.3, 6.3", "Install from uv.lock and fetch starting models", "must pass"),
    (
        5,
        "2.2, 6.5",
        "The train command runs with downloads off, then stops",
        "must pass",
    ),
    (6, "6.3", "The load function returns a Policy or System", "must pass"),
    (7, "6.6", "Export to a portable model file", "must pass"),
    (8, "7.1", "A choice saved within the training time", "measured here"),
    (9, "4.1, 4.2", "Play limits", "measured here"),
)


@dataclass(frozen=True)
class CheckStep:
    """The result of one numbered step of ``check_entry``.

    Attributes
    ----------
    number : int
        The step number, 1 to 9, as in ``check_entry``'s list and the lines it
        prints.
    rules : str
        The rulebook rules this step checks, for example ``"2.3, 6.3"``.
    title : str
        What the step checks, in a few words.
    group : {"must pass", "measured here"}
        Must-pass steps (1 to 7) block ``entry submit`` when they fail.
        Measured-here steps (8 and 9) are decided on our RTX 5090 and never
        block.
    status : {"passed", "failed", "skipped", "not measured"}
        "skipped" means the step did not run, for example because an earlier
        step failed. "not measured" means this machine cannot measure it, for
        example step 9 without a GPU.
    detail : str
        A plain-English explanation: what was found, or what to fix.
    """

    number: int
    rules: str
    title: str
    group: StepGroup
    status: StepStatus
    detail: str


@dataclass(frozen=True)
class PlayLimits:
    """One reading of Rule 4: a system playing itself, timed.

    Attributes
    ----------
    reference : str
        The system that played, as given to ``play_limits``.
    device : str
        JAX's name for the device that played, for example
        ``"NVIDIA GeForce RTX 5090"`` or ``"cpu"``.
    on_gpu : bool
        True when the games ran on a GPU. Only a GPU reading is a measurement
        of Rule 4.
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
        """True when both Rule 4 limits hold, False when one fails.

        None when this is not a GPU reading or the memory is unknown, because
        then Rule 4 was not measured.
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
    """The result of ``check_entry``: every step, by rule number, in two groups.

    Attributes
    ----------
    repository : str
        The folder or URL that was checked.
    commit : str or None
        The full commit SHA that was checked. None when there was no commit.
    machine : str
        This machine's GPU name from ``nvidia-smi``, or "No NVIDIA GPU found".
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
        (no commit, or the tree was not clean).
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
                    "rules": step.rules,
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
    """The checked contents of one ``entry.yaml`` (see _read_entry)."""

    name: str
    train: str
    output: str
    load: str
    entry_type: str
    training_maps: tuple[int, ...]
    validation_maps: tuple[int, ...]
    config: str
    starting_models: tuple[tuple[str, str], ...]


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


def _run(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout: float
) -> tuple[int | None, str, str]:
    """Run a child in its own process group and return (exit code, stdout, stderr).

    The exit code is None when the child ran past ``timeout`` seconds; its whole
    process group is then killed. Output is collected as text.
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
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop_group(process, 0)
        out, err = process.communicate()
        return None, out, err
    return process.returncode, out, err


def _maps(value: object, key: str) -> tuple[int, ...]:
    """Read one list of maps: whole numbers or ranges such as ``"0-41"``, in 0-46."""
    if not isinstance(value, list):
        raise ValueError(f"training_maps.{key} must be a list, for example [42, 43]")
    maps: list[int] = []
    for item in cast(list[object], value):
        if isinstance(item, int) and not isinstance(item, bool):
            maps.append(item)
        elif isinstance(item, str) and re.fullmatch(r"\d+-\d+", item.strip()):
            first, last = (int(part) for part in item.strip().split("-"))
            if first > last:
                raise ValueError(f"training_maps.{key}: {item!r} runs backward")
            maps.extend(range(first, last + 1))
        else:
            raise ValueError(
                f"training_maps.{key}: {item!r} is not a map number or a range "
                'such as "0-41"'
            )
    outside = sorted({m for m in maps if not 0 <= m <= 46})
    if outside:
        raise ValueError(
            f"training_maps.{key} uses maps {outside}; only maps 0 to 46 are "
            "yours to use (Rule 3.4)"
        )
    return tuple(maps)


def _map_text(maps: Sequence[int]) -> str:
    """Write maps briefly: runs of three or more become ranges ("0-41, 44")."""
    parts: list[str] = []
    ordered = sorted(set(maps))
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[end] + 1:
            end += 1
        if end - start >= 2:
            parts.append(f"{ordered[start]}-{ordered[end]}")
        else:
            parts.extend(str(m) for m in ordered[start : end + 1])
        start = end + 1
    return ", ".join(parts) if parts else "none"


def _read_entry(folder: Path) -> _Entry:
    """Read and check ``folder/entry.yaml``; raise ValueError at the first problem.

    The file must be a mapping with exactly the keys ``name``, ``train``,
    ``output``, ``load``, ``entry_type``, ``training_maps``, ``config`` and
    ``starting_models``. ``output`` is a relative path inside the repository.
    ``load`` is ``module:function``. ``training_maps`` holds the lists
    ``training`` (not empty) and ``validation``, each map inside 0-46
    (Rule 3.4). Each starting model is ``{model: ..., sha256: <64 hex>}``
    (Rules 3.1 and 6.3), except a released training-opponent pool, such as
    ``MARL-BattleGrounds/tdm-season-0-results/training-opponents``, which is
    ``{model: ..., commit: <40 hex>}``: a folder has no single SHA-256, and its
    commit fixes every byte. ``starting_models`` keeps (model, digest) pairs,
    the digest lowercased. Needs PyYAML, which the hydra extra installs.
    Raises FileNotFoundError when the file is missing.
    """
    import yaml

    path = folder / _ENTRY_FILE
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{_ENTRY_FILE} must hold the settings as key: value lines")
    data = cast(dict[str, object], raw)
    missing = [key for key in _SETTINGS if key not in data]
    extra = sorted(str(key) for key in data if key not in _SETTINGS)
    if missing or extra:
        raise ValueError(
            f"{_ENTRY_FILE}: missing {missing or 'nothing'}; unknown {extra or 'none'}"
        )
    text: dict[str, str] = {}
    for key in ("name", "train", "output", "load", "entry_type", "config"):
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{_ENTRY_FILE}: {key} must be text that is not empty")
        text[key] = value.strip()
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
        raise ValueError(f"{_ENTRY_FILE}: entry_type must be one of {ENTRY_TYPES}")
    maps = data["training_maps"]
    if not isinstance(maps, dict) or set(cast(dict[str, object], maps)) != {
        "training",
        "validation",
    }:
        raise ValueError(f"{_ENTRY_FILE}: training_maps needs training and validation")
    maps = cast(dict[str, object], maps)
    training, validation = (
        _maps(maps["training"], "training"),
        _maps(maps["validation"], "validation"),
    )
    if not training:
        raise ValueError(f"{_ENTRY_FILE}: training_maps.training is empty")
    models: list[tuple[str, str]] = []
    listed = data["starting_models"]
    if not isinstance(listed, list):
        raise ValueError(f"{_ENTRY_FILE}: starting_models must be a list ([] for none)")
    for item in cast(list[object], listed):
        fields = cast(dict[str, object], item) if isinstance(item, dict) else {}
        model = fields.get("model")
        pool = isinstance(model, str) and _pool(model.strip()) is not None
        key, length = ("commit", 40) if pool else ("sha256", 64)
        digest = fields.get(key)
        if (
            set(fields) != {"model", key}
            or not isinstance(model, str)
            or not model.strip()
            or not isinstance(digest, str)
            or not re.fullmatch(f"[0-9a-fA-F]{{{length}}}", digest)
        ):
            raise ValueError(
                f"{_ENTRY_FILE}: each starting model needs model and sha256 "
                "(64 hex characters), and a training-opponent pool needs model "
                f"and commit (40 hex characters); found {item!r} (Rules 3.1, 6.3)"
            )
        models.append((model.strip(), digest.lower()))
    return _Entry(
        name=text["name"],
        train=text["train"],
        output=output.as_posix(),
        load=text["load"],
        entry_type=text["entry_type"],
        training_maps=training,
        validation_maps=validation,
        config=text["config"],
        starting_models=tuple(models),
    )


_STARTER_TRAIN = (
    "uv run python -m marl_battlegrounds.experiments.train method=mappo "
    "pool=scripted 'validation_opponents=[tdm-alpha,tdm-beta,tdm-gamma]' "
    "output_dir=runs/entry"
)


def _starter_files(name: str, project: str, version: str) -> dict[str, str]:
    """Return the starter's files, by path relative to the entry folder."""
    train = json.dumps(_STARTER_TRAIN)
    return {
        "pyproject.toml": f"""\
# This entry's dependencies. uv.lock pins every version (Rule 6.3).
# On Linux it installs the CUDA 13 build of JAX, which the official run needs.
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
# Your leaderboard entry. `entry check` reads this file and `entry submit`
# copies it into the submission form.

# Your system's name. It becomes the title of your submission.
name: {json.dumps(name)}

# The one command we run to train (Rule 6.3). It keeps its choice saved in
# the output folder below.
train: {train}

# The folder the train command writes, inside this repository.
output: runs/entry

# The function that loads only your chosen entry, written as module:function.
load: my_entry:load

# Learned, Scripted, Hybrid or Language Model (Rule 1.2).
entry_type: learned

# The maps your run trains and validates on. Maps 0 to 46 are yours (Rule 3.4).
training_maps:
  training: ["0-41"]
  validation: [42, 43, 44, 46]

# Your config: a path in this repository, or, as here, the train command.
config: {train}

# Every model your run uses (opponents, teammates and starting weights), each
# with its SHA-256 (Rules 3.1 and 6.3): a released name or a file in this
# repository. A released opponent pool is a folder, so it takes the commit that
# `models download` prints instead. The scripted pool needs none. For example:
#   - model: MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final
#     sha256: <the file's 64-character SHA-256>
#   - model: MARL-BattleGrounds/tdm-season-0-results/training-opponents
#     commit: <the pool's 40-character commit>
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
released Recurrent MAPPO recipe against the scripted pool.

## The Three Commands

Run them in this folder. Commit your work before `entry check`, and push it
before `entry submit`:

```sh
{_STARTER_TRAIN}   # train at home
uv run python -m marl_battlegrounds entry check     # check it like we will
uv run python -m marl_battlegrounds entry submit    # open the submission form
```

`entry check` installs your commit in a fresh folder, trains for 15 minutes,
loads and exports the choice and measures the play limits. `entry submit`
works only after `git push` and a passing check of the same commit.

## Facts About This Starter

- Budget: 250,675,200 training steps plus at most 1,065,600 validation steps
  (31 routine checks of 96 games and up to 3 confirmations of 192 games, at
  most 300 steps a game), under the 255 million of Rule 2.1.
- `retain_all_actors: true` exports a model at every save (every 409,600
  steps), so it keeps one at least every 5,000,000 steps (Rule 6.6).
- It is not timed yet. Season 0's four PPO baselines took 3.6 to 4.3 hours
  each to reach 250 million steps on one RTX 5090, as a 40 million step run
  plus its continuation. Rule 2.1 allows 8 hours.
- The output folder must be new or empty. For a second run at home, change
  `runs/entry` everywhere in `entry.yaml` and in `my_entry/__init__.py`.
""",
    }


def new_entry(path: str | os.PathLike[str], *, name: str | None = None) -> Path:
    """Write a starter entry in a new folder, make it a Git repository and lock it.

    The starter trains the released Recurrent MAPPO recipe against the
    scripted pool and is meant to pass ``entry check`` on an RTX 5090.

    Parameters
    ----------
    path : str or os.PathLike
        The folder to create. It must not exist yet, or be an empty folder.
    name : str or None, default=None
        The system's name, which becomes the submission's title. None uses the
        folder's name.

    Returns
    -------
    Path
        The entry folder, as an absolute path.

    Raises
    ------
    ValueError
        ``path`` is a file or a folder that is not empty, or ``name`` is blank.
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
    It prints the next commands. Nothing is committed. No GPU is used.

    Examples
    --------
    >>> from marl_battlegrounds import entry
    >>> entry.new_entry("my-entry", name="My Team's MAPPO")  # doctest: +SKIP
    """
    folder = Path(path).expanduser().absolute()
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise ValueError(f"{folder} must be a new or empty folder")
    chosen = folder.name if name is None else name
    if not chosen.strip():
        raise ValueError("The entry's name must not be blank")
    if shutil.which("git") is None:
        raise RuntimeError("entry new needs Git. Install Git, then run it again.")
    project = re.sub(r"[^A-Za-z0-9._-]+", "-", folder.name).strip("-._") or "entry"
    for relative, content in _starter_files(
        chosen.strip(), project, _package_version()
    ).items():
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(folder, "init", "--quiet")
    print(f"Wrote a starter entry in {folder}")
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
        '  git add -A && git commit -m "Start my entry"\n'
        "  uv run python -m marl_battlegrounds entry check"
    )
    return folder


def _machine() -> str:
    """Return the first GPU name ``nvidia-smi`` lists, or "No NVIDIA GPU found"."""
    if shutil.which("nvidia-smi") is not None:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            done = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
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


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 as 64 lowercase hex characters."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _released_package(lock: Mapping[str, Any]) -> str:
    """Return the marl-battlegrounds version that a parsed uv.lock installs.

    Rule 6.3: an entry trains and plays on a released marl-battlegrounds from
    PyPI, never on a changed copy. Every package named marl-battlegrounds in
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
                f"marl-battlegrounds=={_package_version()}, then run uv lock "
                "(Rule 6.3)."
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
                    "which is not its file on PyPI. Run uv lock again (Rule 6.3)."
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

    One checker serves one call. ``work`` is the call's temporary folder;
    the clone goes to ``work/entry`` and its Hugging Face cache to
    ``entry/.entry-check-hf``. ``bundle`` is where step 2 keeps a copy of the
    commit (maintainers only), or None.
    """

    def __init__(
        self,
        source: str,
        commit: str | None,
        work: Path,
        train_minutes: float,
        bundle: Path | None,
    ) -> None:
        self.source, self.commit, self.train_minutes = source, commit, train_minutes
        self.work, self.clone, self.bundle = work, work / "entry", bundle
        self.hf_home = self.clone / ".entry-check-hf"
        self.entry: _Entry | None = None
        self.builtin_run = False
        self.choice: Path | None = None
        self.loaded = False
        self.play: PlayLimits | None = None
        self.steps: dict[int, CheckStep] = {}

    def record(self, number: int, status: StepStatus, detail: str) -> None:
        """Save one step's result and print it as one line."""
        _, rules, title, group = _STEPS[number - 1]
        self.steps[number] = CheckStep(number, rules, title, group, status, detail)
        print(f"Step {number} (Rule {rules}) {title}: {status}. {detail}", flush=True)

    def attempt(self, number: int, step: Callable[[], str]) -> bool:
        """Run one must-pass step, record passed or failed, and say if it passed.

        A _StepFailedError message becomes the detail as written; any other error
        is shown with its type, so researcher code cannot crash the check.
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

    def env(self, *, offline: bool) -> dict[str, str]:
        """Return the environment for children in the entry's own environment.

        The clone's .venv comes first on PATH, uv never syncs again, and
        HF_HOME is the check's own fresh cache. Offline adds HF_HUB_OFFLINE
        and UV_OFFLINE. Settings that would point the entry at another
        environment or model cache are dropped (see _own_environment).
        """
        env = _own_environment()
        venv = self.clone / ".venv"
        env.update(
            VIRTUAL_ENV=str(venv),
            PATH=f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}",
            HF_HOME=str(self.hf_home),
            UV_NO_SYNC="1",
        )
        if offline:
            env.update(HF_HUB_OFFLINE="1", UV_OFFLINE="1")
        return env

    def python(
        self,
        code: str,
        *args: str,
        timeout: float,
        extra: Mapping[str, str] | None = None,
    ) -> tuple[int | None, dict[str, Any] | None, str]:
        """Run ``code`` offline with ``uv run --no-sync python -c`` in the clone.

        ``args`` follow the code as ``sys.argv[1:]``; ``extra`` adds
        environment variables. Returns (exit code, or None after ``timeout``
        seconds; the JSON object on the last line of output, or None; the
        last lines of its error output).
        """
        code_out, out, err = _run(
            ["uv", "run", "--no-sync", "python", "-c", code, *args],
            cwd=self.clone,
            env={**self.env(offline=True), **(extra or {})},
            timeout=timeout,
        )
        lines = out.strip().splitlines()
        result: dict[str, Any] | None = None
        if code_out == 0 and lines:
            with contextlib.suppress(json.JSONDecodeError):
                result = json.loads(lines[-1])
        return code_out, result, _tail(err)

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
        """Clone the commit into the temporary folder; keep a bundle if asked."""
        assert self.commit is not None
        _git(self.work, "clone", "--quiet", "--no-checkout", self.source, "entry")
        _git(self.clone, "checkout", "--quiet", "--detach", self.commit)
        if _git(self.clone, "rev-parse", "HEAD") != self.commit:
            raise _StepFailedError(f"The clone is not at {self.commit}.")
        if self.bundle is None:
            return f"Cloned {self.commit} into a temporary folder."
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
        return f"Name {self.entry.name!r}; load {self.entry.load}."

    def step_4(self) -> str:
        """Check the locked marl-battlegrounds, install, then check JAX and models.

        Before installing, uv.lock must exist and install a released
        marl-battlegrounds from PyPI (``_released_package``). Then
        ``uv sync --locked`` runs with the GPU hidden, uv.lock must hold a CUDA
        build of JAX, and every starting model is fetched and its SHA-256 (a
        pool's commit) compared (see fetch).
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
        code, _, err = _run(
            ["uv", "sync", "--locked"],
            cwd=self.clone,
            env=env,
            timeout=_INSTALL_SECONDS,
        )
        seconds = time.monotonic() - started
        if code is None:
            raise _StepFailedError("Installing took more than 1 hour (Rule 2.3).")
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
        for model, digest in self.entry.starting_models:
            self.fetch(model, digest)
        return (
            f"marl-battlegrounds {version} from PyPI. Installed in "
            f"{seconds / 60:.1f} minutes without the GPU; "
            f"{len(self.entry.starting_models)} starting models checked."
        )

    def fetch(self, model: str, digest: str) -> None:
        """Find or download one starting model and compare its digest (step 4).

        A path that exists inside the clone must be a file. A released name
        (``MARL-BattleGrounds/...`` or ``hf://...``) is downloaded with the
        entry's own ``models download`` into the check's HF_HOME, which the
        offline training in step 5 reads. A file's SHA-256 must equal
        ``digest``. A training-opponent pool's commit, read from the address
        ``models download`` prints, must equal ``digest``. Anything else fails.
        """
        local = (self.clone / model).resolve()
        if local.is_relative_to(self.clone.resolve()) and local.exists():
            if not local.is_file():
                raise _StepFailedError(f"{model}: give a file, not a folder.")
            path = local
        elif model.startswith("hf://") or model.lower().startswith(
            "marl-battlegrounds/"
        ):
            code, out, err = _run(
                [
                    *("uv", "run", "--no-sync", "python", "-m", "marl_battlegrounds"),
                    *("models", "download", model),
                ],
                cwd=self.clone,
                env=self.env(offline=False),
                timeout=_INSTALL_SECONDS,
            )
            if _pool(model) is not None:
                pools = [
                    _pool(line.removeprefix(f"{model}: "))
                    for line in out.splitlines()
                    if line.startswith(f"{model}: ")
                ]
                if code != 0 or not pools or pools[-1] is None:
                    raise _StepFailedError(
                        f"models download {model} failed:\n{_tail(err)}"
                    )
                if pools[-1][1] != digest:
                    raise _StepFailedError(
                        f"{model}: its commit is not the one in entry.yaml."
                    )
                return
            found = [
                line.split(":", 1)[1].strip()
                for line in out.splitlines()
                if line.startswith("Local File:")
            ]
            if code != 0 or not found:
                raise _StepFailedError(f"models download {model} failed:\n{_tail(err)}")
            path = Path(found[-1])
        else:
            raise _StepFailedError(
                f"{model}: the check cannot fetch this yet. Use a released name "
                "or a file inside your repository."
            )
        if _sha256(path) != digest:
            raise _StepFailedError(
                f"{model}: its SHA-256 is not the one in entry.yaml."
            )

    def step_5(self) -> str:
        """Train offline for train_minutes, then stop the run's process group."""
        assert self.entry is not None
        log_path = self.work / "train.log"
        calm = True
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                self.entry.train,
                shell=True,
                cwd=self.clone,
                env=self.env(offline=True),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=self.train_minutes * 60.0)
            except subprocess.TimeoutExpired:
                code = None
                calm = _stop_group(process, _STOP_GRACE_SECONDS)
        if code is not None and code != 0:
            output = _tail(log_path.read_text(encoding="utf-8", errors="replace"))
            raise _StepFailedError(
                f"The train command failed (exit code {code}):\n{output}"
            )
        if code == 0:
            return "The train command finished before the time was up."
        return f"Trained for {self.train_minutes:g} minutes, then " + (
            "stopped within 60 seconds of the stop signal."
            if calm
            else "did not stop within 60 seconds of the stop signal and was killed."
        )

    def find_choice(self) -> None:
        """Ask the trainer's own ranking for a built-in run's choice (before step 6).

        A built-in run is an output folder with ``run_details.json``; its
        choice is what ``training.current_choice`` names, or None.
        """
        from marl_battlegrounds.training import current_choice

        assert self.entry is not None
        run = self.clone / self.entry.output
        self.builtin_run = (run / "run_details.json").is_file()
        if self.builtin_run:
            self.choice = current_choice(run)

    def step_6(self) -> str:
        """Call the load function in a fresh process, within 10 minutes."""
        assert self.entry is not None
        code, result, err = self.python(
            _LOAD_CODE, self.entry.load, timeout=_LOAD_SECONDS
        )
        if code is None:
            raise _StepFailedError("Loading took more than 10 minutes (Rule 6.3).")
        if result is None:
            raise _StepFailedError(f"{self.entry.load} failed:\n{err}")
        self.loaded = True
        return f"{self.entry.load} returned a {result['kind']}."

    def step_7(self) -> str:
        """Export the choice (or the loaded Policy); refuse local paths inside it."""
        assert self.entry is not None
        target = self.work / "model.zip"
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

        findings: list[str] = _local_path_findings(target.name, target.read_bytes())
        if findings:
            raise _StepFailedError(
                " ".join(
                    f"{target.name} names {finding}. Train against a released "
                    "name or a model inside your repository, or lock a newer "
                    "marl-battlegrounds, whose export leaves local folders out."
                    for finding in findings
                )
            )
        return f"Exported {target.stat().st_size:,} bytes from {Path(source).name}."

    def step_8(self, machine: str) -> tuple[StepStatus, str]:
        """Say whether a choice was saved within the training time."""
        if self.builtin_run:
            if self.choice is not None:
                return (
                    "passed",
                    f"On {machine}: the run's choice is {self.choice.name}.",
                )
            return "failed", (
                f"On {machine}: the run saved no choice yet. Our RTX 5090 reaches "
                "its first validation sooner."
            )
        if self.loaded:
            return "passed", f"On {machine}: the load function found a saved entry."
        return "failed", f"On {machine}: the load function found no saved entry."

    def step_9(self, machine: str) -> tuple[StepStatus, str]:
        """Measure the play limits in a fresh process with deterministic XLA."""
        assert self.entry is not None
        flags = f"{os.environ.get('XLA_FLAGS', '')} {_DETERMINISTIC}".strip()
        code, result, err = self.python(
            _PLAY_CODE,
            self.entry.load,
            timeout=_PLAY_SECONDS,
            extra={"XLA_FLAGS": flags},
        )
        if code is None:
            return "failed", "The games took more than 30 minutes: far too slow."
        if result is None:
            return "failed", f"The play check failed:\n{err}"
        if not result["on_gpu"]:
            return "not measured", (
                f"No GPU here (JAX found {result['device']}). We measure it on "
                "our RTX 5090."
            )
        self.play = PlayLimits(
            **{key: result[key] for key in PlayLimits.__dataclass_fields__}
        )
        speed, peak = self.play.game_steps_per_second, self.play.peak_memory_bytes
        detail = (
            f"On {result['device']}: {speed:,.0f} game steps per second (Rule 4.1: "
            f"at least {MIN_GAME_STEPS_PER_SECOND:,}); peak "
            + ("unknown" if peak is None else f"{peak / 1e9:.2f} GB")
            + " of GPU memory (Rule 4.2: at most 10 GB)."
        )
        if "RTX 5090" not in machine:
            detail += " Advice only: the limits are decided on our RTX 5090."
        return ("passed" if self.play.within_limits else "failed"), detail

    def after_training(self, machine: str) -> None:
        """Run steps 6 to 9 once step 5 passed.

        Steps 6, 7 and 9 wait (are skipped, not failed) when a built-in run
        saved no choice yet; step 8 then says so. A run whose records cannot
        be read fails step 6.
        """
        try:
            self.find_choice()
        except Exception as error:
            self.record(6, "failed", f"The run's records could not be read: {error}")
            self.record(8, *self.step_8(machine))
            self.skip_rest()
            return
        if self.builtin_run and self.choice is None:
            waiting = "Not checked: no choice was saved yet (see step 8)."
            self.record(6, "skipped", waiting)
            self.record(7, "skipped", waiting)
            self.record(8, *self.step_8(machine))
            self.record(9, "skipped", waiting)
            return
        ok = self.attempt(6, self.step_6) and self.attempt(7, self.step_7)
        self.record(8, *self.step_8(machine))
        if ok:
            self.record(9, *self.step_9(machine))
        self.skip_rest()


def check_entry(
    path: str | os.PathLike[str] = ".",
    *,
    repository: str | None = None,
    commit: str | None = None,
    train_minutes: float = 15.0,
) -> EntryCheck:
    """Check one commit of an entry the way we will, and save the result.

    Steps 1 to 7 must pass on any machine; they block ``entry submit``.
    Steps 8 and 9 are measured here and decided on our RTX 5090; they never
    block. Each step prints one line as it finishes.

    1. The tree is clean and the commit is recorded (Rule 6.3).
    2. A fresh clone of that commit in a temporary folder.
    3. ``entry.yaml`` is present and valid (Rules 3.1, 3.4, 6.3).
    4. ``uv.lock`` installs a released marl-battlegrounds from PyPI, any
       version (checked before installing); ``uv sync --locked`` with the
       GPU hidden, in at most 1 hour; ``uv.lock`` holds a CUDA build of JAX;
       every starting model is fetched into a fresh ``HF_HOME`` in the clone
       and its SHA-256 checked, or a training-opponent pool's commit
       (Rules 2.3, 6.3).
    5. The train command runs for ``train_minutes`` with ``HF_HUB_OFFLINE=1``,
       then gets a stop signal (SIGINT) to its process group, 60 seconds'
       grace, then a kill (Rules 2.2, 6.5).
    6. The load function, in a fresh process, returns a Policy or System
       within 10 minutes (Rule 6.3).
    7. Export to a portable model file: from the actor folder
       ``training.current_choice`` names when the output is a built-in run,
       else from the Policy that the load function returns. The file must
       name no folder on this computer (Rule 6.6).
    8. A choice was saved within the training time (Rule 7.1). When a
       built-in run saved none yet (a slower card may not reach its first
       validation), steps 6, 7 and 9 say so instead of failing.
    9. ``play_limits`` in a fresh process, with XLA's deterministic mode
       (Rules 4.1, 4.2). "Not measured" without a GPU; advice only on any
       card but an RTX 5090.

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
        How long step 5 trains, in minutes (Rule 6.5 uses 15). Must be above 0.

    Returns
    -------
    EntryCheck
        Every step's result. ``passed`` is True when no must-pass step failed.

    Raises
    ------
    ValueError
        Only one of ``repository`` and ``commit`` is given, the commit is not
        a full SHA, or ``train_minutes`` is not above 0.
    RuntimeError
        Git is not installed. Problems inside the steps never raise: they
        become failed steps.
    OSError
        The record could not be written.

    Notes
    -----
    Files: the record is saved as JSON at
    ``.git/marl-bgs-entry-check/<commit>.json`` of the checked repository
    (never committed). With ``repository``, it goes to
    ``<path>/marl-bgs-entry-check/<commit>.json`` beside
    ``<commit>.bundle``, a ``git bundle`` of the commit kept for Rule 6.6.
    Nothing is saved when step 1 fails. The clone, its environment (several
    GB with CUDA) and the run live in a temporary folder (set ``TMPDIR`` to
    choose where) that is removed at the end.

    Processes and network: steps 4 to 9 run the entry's own code in child
    processes inside the clone's locked environment (``uv run --no-sync``).
    Only steps 2 and 4 use the network. Steps 5 to 9 set
    ``HF_HUB_OFFLINE=1`` and ``UV_OFFLINE=1``; this asks tools to stay
    offline but does not seal the network. Steps 5 to 9 may use the GPU.
    Outside code runs on this machine: maintainers run an entry only after
    reading its code (Rule 2.2).

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
    if shutil.which("git") is None:
        raise RuntimeError("entry check needs Git. Install Git, then run it again.")
    folder = Path(path).expanduser().absolute()
    machine = _machine()
    records = None if repository is None else folder / _RECORD_FOLDER
    with tempfile.TemporaryDirectory(
        prefix="marl-bgs-entry-check-", ignore_cleanup_errors=True
    ) as work:
        checker = _Checker(
            repository or str(folder),
            commit,
            Path(work),
            train_minutes,
            None if records is None else records / f"{commit}.bundle",
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
            checker.after_training(machine)
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
            f" Steps {', '.join(map(str, waiting))} were not checked: no choice was "
            "saved yet (see step 8)."
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
    such as ``git@github.com:team/entry.git`` becomes
    ``https://github.com/team/entry``. A user name or token inside the
    address (``https://user:TOKEN@github.com/...``) is removed, so it never
    reaches the printed link or the public issue.
    """
    branches = _git(root, "branch", "-r", "--contains", commit).splitlines()
    branches = [line.strip() for line in branches if "->" not in line]
    if not branches:
        raise ValueError(
            "This commit is on no remote branch. Push it first (git push)."
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
        The link that was printed (and opened). When the filled-in link would
        be longer than ``MAX_LINK_LENGTH`` characters, this is the form with
        only the title filled in, and the other values are printed to copy.

    Raises
    ------
    ValueError
        The tree has uncommitted changes, the commit is on no remote branch
        (push it first), or this commit has no passing ``entry check`` record.
        Also raised when ``entry.yaml`` is not valid.
    RuntimeError
        Git is not installed, or a Git command failed (for example, the
        folder is not inside a Git repository).

    Notes
    -----
    Fills the title (the entry's name) and the text fields ``repository``,
    ``commit``, ``train-command``, ``lock-file`` (``uv.lock``),
    ``load-function``, ``config-and-maps`` (paths and map numbers, not pasted
    files) and ``starting-models``. The team adds its members, the entry type
    and the agreements on the page, and ticks "My repository is private" if
    it is. Nothing is sent until the team presses the button on GitHub; the
    issue's number and time are the public arrival receipt (Rule 7.3).
    Reads only local Git data: it does not fetch from the remote.

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
    commit = _git(root, "rev-parse", "--verify", "HEAD")
    if _git(root, "status", "--porcelain"):
        raise ValueError("Commit your changes first: the tree has uncommitted changes.")
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
    entry = _read_entry(root)
    maps = (
        f"Training maps: {_map_text(entry.training_maps)}\n"
        f"Validation maps: {_map_text(entry.validation_maps)}"
    )
    values = {
        "title": entry.name,
        "repository": repository,
        "commit": commit,
        "train-command": entry.train,
        "lock-file": "uv.lock",
        "load-function": entry.load,
        "config-and-maps": f"Config: {entry.config}\n{maps}",
        "starting-models": "\n".join(
            f"{m} {'commit' if _pool(m) else 'sha256'}:{d}"
            for m, d in entry.starting_models
        )
        or "none",
    }
    link = _issue_link(values)
    if len(link) > MAX_LINK_LENGTH:
        link = _issue_link({"title": entry.name})
        print("The filled-in link would be too long, so copy these into the form:")
        for key in FORM_FIELDS:
            print(f"\n{key}:\n{values[key]}")
    print(f"\nSubmission form for {entry.name} at {commit}:\n{link}")
    print(
        "\nOn the page, add your team members and labs, choose the entry type "
        f'({entry.entry_type}), tick the agreements, and tick "My repository is '
        'private" if it is. Then press Submit new issue.'
    )
    if open_browser:
        open_in_browser(link)
    return link


def play_limits(reference: str, *, games: int = 512, batches: int = 4) -> PlayLimits:
    """Measure Rule 4: a system plays itself; time it and read its peak memory.

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
        ``within_limits`` says whether Rule 4 holds on a GPU.

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
    ``on_gpu`` is False, so it is not a Rule 4 measurement. The peak memory is
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
