"""Build, check and publish the released MARL-BGs models on Hugging Face.

Maintainers run this tool once per season. It turns the converted portable
model files and the saved training runs into one Hugging Face repository per
model, builds the results dataset, checks every file for secrets, uploads, and
makes the release public. Everything goes to the Hugging Face organization
``MARL-BattleGrounds``. Run it from the repository root:

    uv run --no-sync python scripts/dev/publish_models.py build \\
        --models-root CONVERTED --out RELEASE --same-games slightly-different
    uv run --no-sync python scripts/dev/publish_models.py build-dataset --out RELEASE
    uv run --no-sync python scripts/dev/publish_models.py clean \\
        RELEASE/tdm-season-0-results --out CLEAN
    uv run --no-sync python scripts/dev/publish_models.py upload \\
        CLEAN/tdm-season-0-results --private
    uv run --no-sync python scripts/dev/publish_models.py clean \\
        RELEASE/tdm-season-0-* --out CLEAN2 --dataset-commit COMMIT
    uv run --no-sync python scripts/dev/publish_models.py scan CLEAN2
    uv run --no-sync python scripts/dev/publish_models.py upload \\
        CLEAN2/tdm-season-0-recurrent-ippo --private
    uv run --no-sync python scripts/dev/publish_models.py collections --private
    uv run --no-sync python scripts/dev/publish_models.py org-card --release CLEAN2
    uv run --no-sync python scripts/dev/publish_models.py publish
    uv run --no-sync python scripts/dev/publish_models.py cards
    uv run --no-sync python scripts/dev/publish_models.py redo tdm-season-0-qmix

What each command reads and writes:

- ``build`` reads the converted files ``CONVERTED/<repo>/step-XXX_XXX_XXX.zip``
  (only each file's ``model.json``), the two training runs of each model and
  the Season 0 ratings, all under ``artifacts/``. It checks that every save of
  both runs has exactly one matching file. It writes ``RELEASE/<repo>/`` with
  ``README.md`` (the model card), ``config.json``, ``steps.csv``, ``models/``
  (hard links to the converted files, or copies on another file system) and
  ``training-records/0-40m/`` and ``training-records/40m-250m/`` (copies of the
  run records without model folders, learner state or past copies).
- ``build-dataset`` writes ``RELEASE/tdm-season-0-results/``: the Season 0
  tournament run folder and rating tables, ``entrants.csv``, the original
  Season 0 save folders and the training opponents, with a dataset card.
- ``clean`` writes a copy of built folders in which no file names a local
  folder of this computer: paths become released names, dataset addresses and
  ``training-records/`` folders (see public_path), saved descriptions get a new
  ID, and model files keep their program and weights byte for byte. Without
  ``--dataset-commit`` it cleans the dataset without its two games files, which
  name the dataset's own commit; with it, everything else. ``COMMIT`` is the
  commit that uploading ``CLEAN/tdm-season-0-results`` prints. Only the clean
  copies pass ``scan`` and ``upload``, because the built folders still hold
  local paths.
- ``scan`` reads every file in a folder and refuses if any file holds a secret
  token or a local folder path. It names the files, never a secret it found.
- ``upload`` scans one built folder, creates its repository as private if it
  does not exist yet, refuses to change any released file, uploads the folder
  and checks every uploaded file's SHA-256 against the Hub's own record.
- ``collections`` creates the two collections as private. ``org-card`` writes
  the organization card (``README.md`` of the Space ``MARL-BattleGrounds/README``)
  from a built release and removes the Space's default page.
- ``cards`` brings the citation of every card (six models, the dataset and the
  organization) up to date with the README's ``## Citation``. Run it after that
  section changes; keep ``CITATION.cff``'s preferred citation the same (a test
  checks that they agree).
- ``publish`` makes the six model repositories, the dataset, the two
  collections and the organization card public.
- ``redo`` deletes and recreates one ``tdm-season-0-*`` repository, only while
  it is private.

Every Hugging Face command takes ``--dry-run``: it prints the calls it would
make and makes none, not even reads. Every command prints what it did. The tool
never prints a token. It reads ``artifacts/`` and ``README.md``; it writes only
under ``--out`` and to Hugging Face.

Prerequisites: the locked environment with ``huggingface_hub``; the run folders
under ``artifacts/paper-experiments/``; the converted model files; and, for the
Hugging Face commands, ``hf auth login`` with a token that can write to the
organization.

Released model files are never replaced or deleted. ``upload`` refuses to
change any existing file other than ``README.md``, ``steps.csv`` and
``config.json``, and it never deletes. A repository may be deleted and made
again (``redo``) only while it is private. The one exception was the
re-release of 10 October 2026, which replaced every file once with its
``clean`` copy (same programs and weights) and squashed each repository's
history, so no earlier commit still names a local folder. Never squash or
rewrite the dataset's history again: the model files and ``season-0-games``
name its commit ``e8bc31c1f7c2dd679d115074ce953b1307d486d9``.

JAX refresh rule
----------------
JAX promises that an exported program runs on JAX up to 6 months newer than the
JAX that made it. MARL-BGs pins its JAX version, so each package version runs
the JAX its released files were made with. Before moving the package to a JAX
outside that window: load every released file with a newer JAX that is still
inside the window, export a small wrapper that calls it, check its actions on a
recorded batch, and publish the results as new files (new ``format`` rows in
``steps.csv``; old files are never replaced). This needs no original training
code, so it works for entrants' files too.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import re
import shutil
import sys
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from huggingface_hub import (
    DatasetCardData,
    EvalResult,
    HfApi,
    ModelCardData,
    get_token,
)
from huggingface_hub.errors import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import CommitInfo, RepoFile

from marl_battlegrounds.released_models import (
    _GPU_ID,  # pyright: ignore[reportPrivateUsage]
    _local_path_findings,  # pyright: ignore[reportPrivateUsage]
)

ORG = "MARL-BattleGrounds"
DATASET_REPO = "tdm-season-0-results"
ORG_CARD_REPO = f"{ORG}/README"
CONTACT = "t.hawili@sms.ed.ac.uk"
CODE_URL = "https://github.com/arm-2-lab/MARL-BattleGrounds"
# The repository's address before it moved; cleaned text names CODE_URL instead.
OLD_CODE_URL = "https://github.com/oceansystemslab/MARL-BattleGrounds"
GUIDE_URL = f"{CODE_URL}/blob/main/docs/released_models.md"
# The public rulebook; the dataset and organization cards link its leaderboard.
RULES_URL = f"{CODE_URL}/blob/main/docs/tournament_rules.md"
# The day the clean copy replaced every released file, once (see clean).
RE_RELEASE_DAY = "10 October 2026"
DATASET_URL = f"https://huggingface.co/datasets/{ORG}/{DATASET_REPO}"
PORTABLE_FORMAT = "marl-bgs-portable-model"
FILE_FORMAT = "portable-model-1"
SEASON = 0

_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
# The repository's README; its "## Citation" section is the one citation source.
_README = Path(__file__).resolve().parents[2] / "README.md"
_SEARCH = Path("paper-experiments/hyperparameter-search")
_RATINGS = _SEARCH / "results/tournament-250m/ratings.csv"
_RATING_TABLES = _SEARCH / "results/tournament-250m"
_TOURNAMENT = Path(
    "paper-experiments/tournament-analysis-season-0/full-metrics-timed/out/replay/"
    "tournament_20261007_095226_96f9aa8c5e85"
)
_OPPONENTS_ZIP = Path("website-20260930/site/downloads/season-0/training-opponents.zip")
FIRST_RECORDS = "0-40m"
SECOND_RECORDS = "40m-250m"
# Run folder entries that are not training records: model folders, learner
# state, past copies and the run lock.
_NOT_RECORDS = frozenset(("actors", "checkpoints", "past_copies", ".training.lock"))
# Files that later commits may change. Every other uploaded file is final.
MUTABLE_FILES = frozenset(("README.md", "steps.csv", "config.json"))
STEPS_COLUMNS = (
    "step",
    "kind",
    "file",
    "bytes",
    "sha256",
    "original_checkpoint_id",
    "actor_digest",
    "validation_games",
    "validation_point_margin",
    "validation_kills",
    "validation_deaths",
    "format",
)
_LARGE_FILE = 50_000_000
_ACTOR_FOLDER = re.compile(r"_actor_step_(\d+)_([0-9a-f]{64})$")
_CONFIG_CLASSES = {
    "mappo": ("PPOConfig", "ppo"),
    "ippo": ("PPOConfig", "ppo"),
    "ff_mappo": ("PPOConfig", "ppo"),
    "ff_ippo": ("PPOConfig", "ppo"),
    "qmix": ("QMIXConfig", "qmix"),
    "pqn_vdn": ("PQNConfig", "pqn"),
}
SAME_GAMES = {
    "identical": (
        "Every file holds its save's exact weights and the model's one decision "
        "program (checked for every save). In the release check, the Season 0 "
        "save's file matched the original save folder in all 1,080 games against "
        "TDM-Alpha, TDM-Beta and TDM-Gamma on test maps 47-51 (one RTX 5090, "
        "normal GPU mode, same seeds)."
    ),
    "slightly-different": (
        "Every file holds its save's exact weights and the model's one decision "
        "program (checked for every save). Its games can differ slightly from "
        "the original save's, about as much as two GPU runs of the original "
        "differ from each other."
    ),
}
_SAME_INPUTS = (
    "On CPU, or on GPU in deterministic mode, the same file, inputs and settings "
    "replay the same games. The Season 0 tournament played the original save "
    "folders, which the results dataset keeps under `season-0-originals/` with "
    "their weights unchanged."
)
# One pattern per kind of secret. Each starts with fixed text, which lets the
# regular expression engine skip quickly through large files.
_SECRETS = {
    "Hugging Face token pattern": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "GitHub token pattern": re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}"),
    "GitHub fine-grained token pattern": re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    "PyPI token pattern": re.compile(rb"pypi-AgEI[A-Za-z0-9_-]{16,}"),
    "AWS access key pattern": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private key header": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
_SAVED_TOKEN = "the saved Hugging Face token"


class PublishError(Exception):
    """A check failed, so the command stopped before doing anything unsafe."""


_ARXIV_ID = re.compile(r"(?:arXiv:|arxiv\.org/abs/)(\d{4}\.\d{4,5})")
_CITE_HEADING = "## Cite This Work"


def paper_citation(readme: str) -> tuple[str, str | None]:
    """Return the paper's BibTeX entry and arXiv id from the README's citation.

    Parameters
    ----------
    readme : str
        The text of the repository's ``README.md``. Its ``## Citation`` section
        is the one source of the citation the cards show.

    Returns
    -------
    tuple of (str, str or None)
        The first ``bibtex`` code block of that section, without the fences,
        and the arXiv id (such as ``"2610.01234"``) when the entry names one,
        else None.

    Raises
    ------
    PublishError
        The README has no ``## Citation`` section with a BibTeX block, or the
        block cites the software (``@software``) instead of the paper.
    """
    heading = re.search(r"^## Citation\b", readme, re.MULTILINE)
    start = heading.start() if heading else -1
    end = readme.find("\n## ", start + 1) if start >= 0 else -1
    section = readme[start : end if end >= 0 else len(readme)] if start >= 0 else ""
    match = re.search(r"```bibtex\n(.*?)\n```", section, re.DOTALL)
    if match is None:
        raise PublishError("README.md has no ## Citation section with a BibTeX block")
    entry = match.group(1).strip()
    if entry.lower().startswith("@software"):
        raise PublishError("README.md cites the software; cards cite the paper")
    found = _ARXIV_ID.search(entry)
    return entry, found.group(1) if found else None


def citation_section(entry: str, arxiv: str | None) -> str:
    """Return the cards' "Cite This Work" section for one BibTeX entry.

    Parameters
    ----------
    entry : str
        The paper's BibTeX entry (see paper_citation).
    arxiv : str or None
        The paper's arXiv id. When given, a link to its arXiv page comes first.

    Returns
    -------
    str
        The section, starting with its heading and ending with one newline.
    """
    link = f"Paper: https://arxiv.org/abs/{arxiv}\n\n" if arxiv else ""
    return (
        f"{_CITE_HEADING}\n\nIf you use this work, please cite the paper.\n\n"
        f"{link}```bibtex\n{entry}\n```\n"
    )


def with_citation(card: str, entry: str, arxiv: str | None) -> str:
    """Return a card with its citation section and arXiv tag brought up to date.

    Parameters
    ----------
    card : str
        A model card, the dataset card or the organization card, as uploaded.
    entry : str
        The paper's BibTeX entry.
    arxiv : str or None
        The paper's arXiv id, or None before it has one.

    Returns
    -------
    str
        The same card with its ``## Cite This Work`` section replaced (up to the
        next ``## `` heading) and, when ``arxiv`` is given and the YAML header
        has a ``tags:`` list, an ``arxiv:<id>`` tag added. Hugging Face lists
        a repository with that tag on the paper's page.

    Raises
    ------
    PublishError
        The card has no ``## Cite This Work`` section.
    """
    start = card.find(_CITE_HEADING)
    if start < 0:
        raise PublishError(f"The card has no {_CITE_HEADING!r} section")
    end = card.find("\n## ", start + 1)
    after = card[end + 1 :] if end >= 0 else ""
    card = (
        card[:start] + citation_section(entry, arxiv) + ("\n" + after if after else "")
    )
    if arxiv and card.startswith("---\n") and f"arxiv:{arxiv}" not in card:
        header_end = card.find("\n---", 4)
        tags = card.find("\ntags:\n", 0, header_end)
        if tags >= 0:
            at = tags + len("\ntags:\n")
            card = card[:at] + f"- arxiv:{arxiv}\n" + card[at:]
    return card


@dataclass(frozen=True)
class Model:
    """One released model: its repository name and its two training runs.

    Attributes
    ----------
    repo : str
        Repository name inside the organization, such as
        ``"tdm-season-0-recurrent-ippo"``.
    first_run : str
        The run that trained from step 0 to 40,108,032, relative to
        ``artifacts/paper-experiments/hyperparameter-search``.
    second_run : str
        The run that continued the first run to the end of training, relative
        to the same folder. Both runs hold the save at 40,108,032; the release
        uses the first run's copy.
    """

    repo: str
    first_run: str
    second_run: str

    @property
    def case(self) -> str:
        """Return the study case name that both runs and the ratings share."""
        return PurePosixPath(self.first_run).parent.name


def _model(repo: str, first: str, second: str, case: str) -> Model:
    """Return the Model whose two runs are ``<study>/study/cases/<case>/run``."""
    return Model(
        repo, f"{first}/study/cases/{case}/run", f"{second}/study/cases/{case}/run"
    )


# The six Season 0 models, in Season 0 rank order (the collections keep it).
MODELS = (
    _model(
        "tdm-season-0-recurrent-ippo",
        "shapes-and-ppo",
        "250m",
        "ippo-reference--seed-20261001",
    ),
    _model(
        "tdm-season-0-pqn-vdn",
        "q-settings",
        "250m",
        "pqn_vdn-b768-t128-reward-x10--seed-12259871",
    ),
    _model(
        "tdm-season-0-recurrent-mappo",
        "shapes-and-ppo",
        "250m",
        "mappo-reference--seed-20261002",
    ),
    _model(
        "tdm-season-0-qmix",
        "q-settings",
        "250m",
        "qmix-b768-t32-lr-1e-4-target-faster--seed-12259871",
    ),
    _model(
        "tdm-season-0-feedforward-ippo",
        "shapes-and-ppo",
        "250m",
        "ff_ippo-reference--seed-20261001",
    ),
    _model(
        "tdm-season-0-feedforward-mappo",
        "ff-mappo-levers",
        "ff-mappo-250m",
        "ff_mappo-team-reward--seed-12259871",
    ),
)
MODEL_REPOS = tuple(model.repo for model in MODELS)
type _Item = tuple[str, Literal["model", "dataset"]]
_MODEL_ITEMS: tuple[_Item, ...] = tuple(
    (f"{ORG}/{repo}", "model") for repo in MODEL_REPOS
)
# (title, description, items). Hugging Face keeps collection descriptions short.
COLLECTIONS: tuple[tuple[str, str, tuple[_Item, ...]], ...] = (
    (
        "Season 0 Big Nine (TDM)",
        "Season 0 Team Deathmatch: six trained models and the results. TDM-Alpha, "
        "TDM-Beta and TDM-Gamma ship inside the package, so the field is nine.",
        (*_MODEL_ITEMS, (f"{ORG}/{DATASET_REPO}", "dataset")),
    ),
    (
        "General Pool (TDM)",
        "Every released Team Deathmatch entrant. Use any of them by name as an "
        "opponent, a teammate or a starting point.",
        _MODEL_ITEMS,
    ),
)


@dataclass(frozen=True)
class Save:
    """One saved actor folder of a training run.

    Attributes
    ----------
    step : int
        Training step of the save, in environment transitions (one game
        advancing one tick is one transition).
    records : str
        Which run made it: ``FIRST_RECORDS`` ("0-40m") or ``SECOND_RECORDS``
        ("40m-250m").
    folder : Path
        The actor folder under the run's ``actors/``.
    checkpoint_id : str
        The save's id in the run records: the folder name's suffix, also used
        by ``validation_results.json`` and the Season 0 tournament tables.
    export_id : str
        The ``checkpoint_id`` written at the top of the folder's
        ``actor_details.json``.
    actor_digest : str
        The ``actor_digest`` of ``actor_details.json``.
    """

    step: int
    records: str
    folder: Path
    checkpoint_id: str
    export_id: str
    actor_digest: str


@dataclass(frozen=True)
class ModelFile:
    """One converted portable model file that matched a save.

    Attributes
    ----------
    save : Save
        The save the file was converted from.
    path : Path
        The converted file under ``--models-root``.
    size : int
        File size in bytes.
    sha256 : str
        SHA-256 of the whole file, as lowercase hex.
    """

    save: Save
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class Validation:
    """One routine validation check of one save, from the run records.

    Attributes
    ----------
    games : int
        Games played in the check.
    point_margin : float or None
        Our points minus the opponent's points per game. None when the saved
        check lacks points.
    kills : float or None
        Our kills per game, or None when missing.
    deaths : float or None
        Our deaths (the opponent's kills) per game, or None when missing.

    Notes
    -----
    The three per-game values average the map and opponent groups with equal
    weight, as ``marl_battlegrounds.training.analysis`` defines them.
    """

    games: int
    point_margin: float | None
    kills: float | None
    deaths: float | None


@dataclass(frozen=True)
class Panel:
    """The fixed validation panel both runs of one model used.

    Attributes
    ----------
    members : tuple of tuple of (str, str or None)
        Each opponent's name and, for a saved model, its folder name (for
        example ``"recurrent-mappo-combat-support-pooled-step-250675200"``);
        None for a built-in name such as ``"tdm-alpha"``.
    maps : tuple of int
        Validation map ids, ascending.
    games : tuple of int
        Distinct games-per-check values of the routine checks, ascending.
        Normally one value.
    """

    members: tuple[tuple[str, str | None], ...]
    maps: tuple[int, ...]
    games: tuple[int, ...]


@dataclass(frozen=True)
class Entrant:
    """One learned Season 0 entrant's row of the tournament ratings.

    Attributes
    ----------
    entrant : str
        The tournament's entrant name, such as
        ``"IPPO at step 250,675,200 (run)"``.
    label : str
        Display name, such as ``"Recurrent IPPO"``.
    method : str
        Training method key, such as ``"ippo"`` or ``"pqn_vdn"``.
    seed : int
        Training seed.
    recipe : str
        Training recipe name.
    step : int
        The save that played Season 0.
    folder_name : str
        That save's actor folder name.
    rank : int
        Season 0 rank, 1 being best.
    field : int
        Number of Season 0 entrants.
    elo, elo_low, elo_high : float
        Season 0 Elo and its 95% interval.
    tier : str
        Season 0 tier, such as ``"Platinum 4"``.
    """

    entrant: str
    label: str
    method: str
    seed: int
    recipe: str
    step: int
    folder_name: str
    rank: int
    field: int
    elo: float
    elo_low: float
    elo_high: float
    tier: str


def step_text(step: int) -> str:
    """Return a step as it follows ``@`` in a released name.

    Parameters
    ----------
    step : int
        Nonnegative training step.

    Returns
    -------
    str
        Digits in groups of three without leading zeros, such as
        ``"99_950_592"`` or ``"250_675_200"``.
    """
    return f"{step:_d}"


def model_file_name(step: int) -> str:
    """Return the released file name of the save at ``step``.

    Parameters
    ----------
    step : int
        Nonnegative training step.

    Returns
    -------
    str
        Nine digits in groups of three, padded with zeros so names sort by
        step, such as ``"step-000_409_600.zip"`` or ``"step-100_139_008.zip"``.
        Steps of a billion or more use more digits
        (``"step-1_234_567_890.zip"``).
    """
    return f"step-{step:011_d}.zip"


def _repo(name: str) -> tuple[str, str]:
    """Return the full repository id and type of a repository name.

    Parameters
    ----------
    name : str
        A name inside the organization, with or without the ``ORG/`` prefix.

    Returns
    -------
    tuple of (str, str)
        ``("MARL-BattleGrounds/<name>", "dataset")`` for the results dataset,
        else the same id with ``"model"``.
    """
    name = name.removeprefix(f"{ORG}/")
    return f"{ORG}/{name}", "dataset" if name == DATASET_REPO else "model"


def _size(value: int) -> str:
    """Return a byte count in decimal units, such as ``"366.6 MB"``."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value} B"


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file as lowercase hex, reading 1 MiB at a time."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_id(path: Path) -> str:
    """Return the Git blob id (SHA-1 of header and bytes) Git gives a file."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object) -> dict[str, object]:
    """Return a parsed JSON object with text keys, or ``{}`` for anything else."""
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in cast(dict[object, object], value).items()}


def _list(value: object) -> list[object]:
    """Return a parsed JSON list, or ``[]`` for anything else."""
    return list(cast(list[object], value)) if isinstance(value, list) else []


def _number(value: object) -> float | None:
    """Return a real number as float, or None for anything else (Booleans too)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load(path: Path) -> object:
    """Parse one JSON file.

    Raises
    ------
    PublishError
        The file is missing or is not JSON.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise PublishError(f"Cannot read {path}: {error}") from error


def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from a file.

    Raises
    ------
    PublishError
        The file is missing, is not JSON or does not hold an object.
    """
    value = _load(path)
    result = _object(value)
    if not isinstance(value, dict):
        raise PublishError(f"{path} does not hold a JSON object")
    return result


def _run_saves(run: Path, records: str) -> dict[int, Save]:
    """Read every saved actor folder of one run.

    Parameters
    ----------
    run : Path
        A training run folder with an ``actors/`` folder.
    records : str
        The label the saves get (``FIRST_RECORDS`` or ``SECOND_RECORDS``).

    Returns
    -------
    dict of int to Save
        One Save per actor folder, keyed by step.

    Raises
    ------
    PublishError
        ``actors/`` is missing, a folder name is not a saved actor name, its
        ``actor_details.json`` names another save, or two folders share a step.
    """
    actors = run / "actors"
    if not actors.is_dir():
        raise PublishError(f"No actors/ folder in {run}")
    saves: dict[int, Save] = {}
    for folder in sorted(actors.iterdir()):
        match = _ACTOR_FOLDER.search(folder.name)
        if match is None or not folder.is_dir():
            raise PublishError(f"Not a saved actor folder: {folder}")
        step, identifier = int(match[1]), match[2]
        details = _read_json(folder / "actor_details.json")
        metadata = _object(details.get("metadata"))
        if (
            metadata.get("checkpoint_id") != identifier
            or metadata.get("env_steps") != step
        ):
            raise PublishError(f"{folder} does not describe its own save")
        if step in saves:
            raise PublishError(f"Two saves at step {step:,} in {run}")
        saves[step] = Save(
            step,
            records,
            folder,
            identifier,
            str(details.get("checkpoint_id")),
            str(details.get("actor_digest")),
        )
    return saves


def read_saves(first_run: Path, second_run: Path) -> dict[int, Save]:
    """Return every released save of one model, from both of its runs.

    Parameters
    ----------
    first_run : Path
        The run from step 0 to 40,108,032.
    second_run : Path
        The run that continued it. Its saves must start at or after the first
        run's last save.

    Returns
    -------
    dict of int to Save
        Saves keyed by step, ascending. The save both runs hold at the first
        run's last step (40,108,032 for Season 0) comes from the first run; the
        second run's copy, which has another id, is left out.

    Raises
    ------
    PublishError
        A run cannot be read (see ``_run_saves``) or the second run has a save
        before the first run's last save.
    """
    saves = _run_saves(first_run, FIRST_RECORDS)
    later = _run_saves(second_run, SECOND_RECORDS)
    if saves and later:
        boundary = max(saves)
        early = sorted(step for step in later if step < boundary)
        if early:
            raise PublishError(
                f"{second_run} has saves before {boundary:,}: {early[:3]}"
            )
        later.pop(boundary, None)
    saves.update(later)
    return dict(sorted(saves.items()))


def read_validation(run: Path) -> dict[str, Validation]:
    """Return each validated save's routine check from one run's records.

    Parameters
    ----------
    run : Path
        A training run folder. A missing ``validation_results.json`` gives an
        empty result.

    Returns
    -------
    dict of str to Validation
        Keyed by the save's run-record id (``Save.checkpoint_id``). Only
        complete routine checks count; confirmation games stay in the records.

    Raises
    ------
    PublishError
        One save has two routine checks with different results.
    ValueError
        ``validation_results.json`` is malformed (raised by the analysis
        owner).

    Notes
    -----
    The summary comes from ``marl_battlegrounds.training.analysis``, the owner
    of the per-game margin, kills and deaths of a check.
    """
    from marl_battlegrounds.training.analysis import (
        _progress_checks,  # pyright: ignore[reportPrivateUsage]
    )

    results: dict[str, Validation] = {}
    for check in _progress_checks(run / "validation_results.json"):
        games = sum(
            int(_number(_object(row).get("games")) or 0)
            for row in _list(check.get("opponents"))
        )
        value = Validation(
            games,
            _number(check.get("margin")),
            _number(check.get("kills_for")),
            _number(check.get("kills_against")),
        )
        identifier = str(check.get("checkpoint_id"))
        if results.get(identifier, value) != value:
            raise PublishError(f"Save {identifier} has two different routine checks")
        results[identifier] = value
    return results


def read_panel(first_run: Path, second_run: Path) -> Panel:
    """Return the validation panel both runs of one model used.

    Parameters
    ----------
    first_run : Path
        The first run; its ``validation_panel/panel.json`` names the panel.
    second_run : Path
        The run that continued it.

    Returns
    -------
    Panel
        The panel's members, the validation maps and the games per routine
        check, read from the records.

    Raises
    ------
    PublishError
        The panel file is missing, or a saved validation result of either run
        used another panel.
    """
    panel = _read_json(first_run / "validation_panel/panel.json")
    digest = panel.get("panel_digest")
    members: list[tuple[str, str | None]] = []
    for item in _list(panel.get("members")):
        member = _object(item)
        reference = str(member.get("reference"))
        folder = PurePosixPath(reference).name if "/" in reference else None
        members.append((str(member.get("name")), folder))
    maps: set[int] = set()
    games: set[int] = set()
    for run in (first_run, second_run):
        path = run / "validation_results.json"
        for item in _list(_load(path)):
            result = _object(item)
            if result.get("panel_digest") != digest:
                raise PublishError(f"{path} holds results from another panel")
            if result.get("purpose") == "routine":
                maps.update(
                    int(_number(value) or 0) for value in _list(result.get("maps"))
                )
                games.add(int(_number(result.get("games")) or 0))
    if not members or not maps:
        raise PublishError(f"No validation panel or maps for {first_run}")
    return Panel(tuple(members), tuple(sorted(maps)), tuple(sorted(games)))


def read_entrant(ratings: Path, case: str) -> Entrant:
    """Return one learned entrant's Season 0 row.

    Parameters
    ----------
    ratings : Path
        The Season 0 ``ratings.csv``.
    case : str
        The model's study case name, as in the ``case_id`` column.

    Returns
    -------
    Entrant
        The row's facts. ``field`` counts every row of the table.

    Raises
    ------
    PublishError
        The table cannot be read or does not have exactly one row for ``case``.
    """
    try:
        with ratings.open(newline="") as file:
            rows = list(csv.DictReader(file))
    except OSError as error:
        raise PublishError(f"Cannot read {ratings}: {error}") from error
    found = [row for row in rows if row.get("case_id") == case]
    if len(found) != 1:
        raise PublishError(f"{ratings} has {len(found)} rows for {case}")
    row = found[0]
    return Entrant(
        entrant=row["entrant"],
        label=row["label"],
        method=row["method"],
        seed=int(row["seed"]),
        recipe=row["recipe"],
        step=int(row["checkpoint_env_steps"]),
        folder_name=PurePosixPath(row["reference"]).name,
        rank=int(row["rank"]),
        field=len(rows),
        elo=float(row["elo"]),
        elo_low=float(row["elo_ci_low"]),
        elo_high=float(row["elo_ci_high"]),
        tier=row["tier"],
    )


def _entered_save(saves: Mapping[int, Save], entrant: Entrant) -> Save:
    """Return the save that played Season 0, checked against the ratings.

    Raises
    ------
    PublishError
        No save has the entered step, or its folder is not the rated folder.
    """
    save = saves.get(entrant.step)
    if save is None or save.folder.name != entrant.folder_name:
        raise PublishError(
            f"The rated save {entrant.folder_name} is not among the run saves"
        )
    return save


def match_model_files(
    folder: Path, saves: Mapping[int, Save], method: str
) -> list[ModelFile]:
    """Match every converted file of one model to exactly one save.

    Parameters
    ----------
    folder : Path
        ``--models-root/<repo>``, holding ``step-XXX_XXX_XXX.zip`` files.
    saves : mapping of int to Save
        The model's released saves (see ``read_saves``).
    method : str
        The model's training method; every file's ``model.json`` must agree.

    Returns
    -------
    list of ModelFile
        One per save, ascending by step.

    Raises
    ------
    PublishError
        A file is not a format 1 portable model, is misnamed, names another
        method, step, id or actor digest, or a save has no file.

    Notes
    -----
    Only ``model.json`` is read from each ZIP. Its ``checkpoint_id`` may be
    either id of the save (the run-record id or the ``actor_details.json``
    id); its ``actor_digest`` must equal the save's.
    """
    files: dict[int, ModelFile] = {}
    for path in sorted(folder.glob("*.zip")):
        try:
            with zipfile.ZipFile(path) as archive:
                model = _object(json.loads(archive.read("model.json")))
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
            raise PublishError(f"Cannot read model.json in {path}: {error}") from error
        get = model.get
        step = get("training_step")
        if get("format") != PORTABLE_FORMAT or get("format_version") != 1:
            raise PublishError(f"{path} is not a {PORTABLE_FORMAT} version 1 file")
        if not isinstance(step, int) or path.name != model_file_name(step):
            raise PublishError(f"{path} does not hold the step its name says")
        save = saves.get(step)
        if save is None:
            raise PublishError(f"{path}: no run save at step {step:,}")
        if get("method") != method:
            raise PublishError(f"{path}: method {get('method')!r}, expected {method!r}")
        if get("checkpoint_id") not in (save.checkpoint_id, save.export_id):
            raise PublishError(f"{path} was not converted from {save.folder}")
        if get("actor_digest") != save.actor_digest:
            raise PublishError(f"{path}: actor_digest differs from {save.folder}")
        files[step] = ModelFile(save, path, path.stat().st_size, _sha256(path))
    missing = [step for step in saves if step not in files]
    if missing:
        raise PublishError(
            f"{folder}: {len(missing)} saves have no file, first "
            f"{[model_file_name(step) for step in missing[:3]]}"
        )
    return [files[step] for step in sorted(files)]


def _link_or_copy(source: Path, target: Path) -> bool:
    """Hard-link ``source`` to ``target``, or copy it when linking fails.

    Returns
    -------
    bool
        True for a hard link, False for a copy (for example across file
        systems).
    """
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)
        return False
    return True


def copy_records(run: Path, target: Path) -> list[tuple[str, int]]:
    """Copy one run's training records, leaving out models and learner state.

    Parameters
    ----------
    run : Path
        A training run folder. It is only read.
    target : Path
        A new folder for the copies.

    Returns
    -------
    list of tuple of (str, int)
        Every copied file's path inside ``target`` (POSIX form) and size.

    Notes
    -----
    Left out: ``actors/`` (model folders), ``checkpoints/`` (learner state),
    ``past_copies/`` and the lock file ``.training.lock``. Everything else is
    copied with its timestamps.
    """

    def skip(folder: str, names: list[str]) -> list[str]:
        """Name the top-level entries of ``run`` that are not records."""
        return [name for name in names if folder == str(run) and name in _NOT_RECORDS]

    shutil.copytree(run, target, ignore=skip)
    return [
        (path.relative_to(target).as_posix(), path.stat().st_size)
        for path in sorted(target.rglob("*"))
        if path.is_file()
    ]


def _write_csv(
    path: Path, columns: Sequence[str], rows: Iterable[Sequence[object]]
) -> None:
    """Write a CSV table with a header row and Unix line endings."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _cell(value: float | None) -> str:
    """Return a float as its shortest exact text, or an empty cell for None."""
    return "" if value is None else repr(value)


def steps_rows(
    files: Sequence[ModelFile], validation: Mapping[str, Validation]
) -> list[list[object]]:
    """Return the ``steps.csv`` rows, one per model file, in step order.

    Parameters
    ----------
    files : sequence of ModelFile
        The matched files (see ``match_model_files``).
    validation : mapping of str to Validation
        Routine checks keyed by run-record id (see ``read_validation``).

    Returns
    -------
    list of list
        Rows in ``STEPS_COLUMNS`` order. ``original_checkpoint_id`` is the
        save's run-record id. The four validation cells are empty when the run
        did not check that save.
    """
    rows: list[list[object]] = []
    for item in sorted(files, key=lambda item: item.save.step):
        check = validation.get(item.save.checkpoint_id)
        rows.append(
            [
                item.save.step,
                "model",
                f"models/{item.path.name}",
                item.size,
                item.sha256,
                item.save.checkpoint_id,
                item.save.actor_digest,
                "" if check is None else check.games,
                "" if check is None else _cell(check.point_margin),
                "" if check is None else _cell(check.kills),
                "" if check is None else _cell(check.deaths),
                FILE_FORMAT,
            ]
        )
    return rows


def _members_text(panel: Panel) -> str:
    """Return the panel's members as one plain-English list."""
    names = [
        f"`{name}` (the model `{folder}`, released in the results dataset under "
        "`training-opponents/models/`)"
        if folder
        else f"`{name}`"
        for name, folder in panel.members
    ]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


def _numbers_text(values: Sequence[int]) -> str:
    """Return numbers as ``"42, 43, 44 and 46"``."""
    texts = [str(value) for value in values]
    return texts[0] if len(texts) == 1 else ", ".join(texts[:-1]) + f" and {texts[-1]}"


def model_card(
    model: Model,
    entrant: Entrant,
    saves: Mapping[int, Save],
    panel: Panel,
    same_games: str,
    input_scale: float,
    repo_bytes: int,
    citation: tuple[str, str | None],
) -> str:
    """Return the model card (``README.md``) of one model repository.

    Parameters
    ----------
    model : Model
        The model.
    entrant : Entrant
        Its Season 0 ratings row.
    saves : mapping of int to Save
        Its released saves, ascending by step.
    panel : Panel
        The validation panel behind the ``steps.csv`` validation columns.
    same_games : {"identical", "slightly-different"}
        Which "same games" claim the conversion check supported.
    input_scale : float
        The model's input scale; starting training from it needs the same.
    repo_bytes : int
        Total size in bytes of the repository's other files, for the
        download note.
    citation : tuple of (str, str or None)
        The paper's BibTeX entry and arXiv id (see paper_citation).

    Returns
    -------
    str
        Hugging Face YAML metadata, then the card in plain English, ending
        with "Cite This Work" and "License and Contact".

    Raises
    ------
    KeyError
        ``same_games`` or the method is unknown.
    """
    name = f"{ORG}/{model.repo}"
    steps = list(saves)
    first, last = steps[0], steps[-1]
    boundary = max(
        step for step, save in saves.items() if save.records == FIRST_RECORDS
    )
    near = max((step for step in steps if step <= 100_000_000), default=last)
    config_class, config_field = _CONFIG_CLASSES[entrant.method]
    claim = SAME_GAMES[same_games]
    metadata = ModelCardData(
        license="apache-2.0",
        library_name="marl-battlegrounds",
        pipeline_tag="reinforcement-learning",
        tags=[
            "reinforcement-learning",
            "multi-agent-reinforcement-learning",
            "jax",
            "marl-battlegrounds",
            "team-deathmatch",
            *([f"arxiv:{citation[1]}"] if citation[1] else []),
        ],
        model_name=model.repo,
        eval_results=[
            EvalResult(
                task_type="reinforcement-learning",
                task_name="Team Deathmatch",
                dataset_type=f"{ORG}/{DATASET_REPO}",
                dataset_name="TDM Season 0",
                metric_type="elo",
                metric_value=round(entrant.elo),
                metric_name="Season 0 Elo",
            )
        ],
    ).to_yaml()
    games = _numbers_text(panel.games)
    return f"""---
{metadata}
---

# {entrant.label} (Team Deathmatch, Season 0)

This is the {entrant.label} model that played Season 0 of Team Deathmatch in
[MARL-BattleGrounds]({CODE_URL}) (MARL-BGs), a JAX benchmark where two teams of
five agents (Mage, Warrior, Hunter, Rogue and Priest) fight. This repository
holds a portable model file for every save of its training, and its training
records.

- **Method:** {entrant.label} (`{entrant.method}`)
- **Training Recipe:** `{entrant.recipe}`
- **Training Seed:** {entrant.seed}
- **Saves:** {len(steps):,}, from step {first:,} to step {last:,}
- **Season 0 Save:** Step {entrant.step:,} (`models/{model_file_name(entrant.step)}`)
- **Season 0 Rank:** {entrant.rank} of {entrant.field}
- **Season 0 Elo:** {round(entrant.elo):,} (95% interval {round(entrant.elo_low):,} \
to {round(entrant.elo_high):,}), tier {entrant.tier}

A step is one environment transition: one game moving forward one tick.

## Use It

```python
import marl_battlegrounds as marl_bgs

name = "{name}"
system = marl_bgs.load_method(name)  # the save that played Season 0
result = marl_bgs.evaluate(system, "tdm-alpha", num_episodes=20)
```

Add `@` to the name to pick another save: `@final` is the last save (step
{last:,}), `@100M` is the nearest save at or before 100,000,000 steps (step
{near:,}), and `@{step_text(near)}` is exactly that save. A step with no save
gives an error that lists the nearest saves.

A name works anywhere MARL-BGs takes a model, such as an opponent, a teammate,
an evaluation or the BattleClient. Each file downloads once and is checked
against `steps.csv` before use. The examples below reuse `marl_bgs` and `name`.
The [Released Models guide]({GUIDE_URL}) shows the whole workflow.

### Train Against a Ladder of Its Saves

```python
from marl_battlegrounds import training

ladder = [f"{{name}}@{{millions}}M" for millions in (50, 100, 150, 200)]
config = training.TrainConfig(
    method="mappo",
    opponents={{"self": 0.2, **dict.fromkeys(ladder, 0.2)}},
    validation_opponents=(name, "tdm-alpha"),
)
training.train(config, output_dir="runs/ladder")
```

To play a mix of saves in your own evaluation, use
`marl_bgs.pool({{f"{{name}}@50M": 1.0, f"{{name}}@final": 1.0}})`.

### Start Training From It

```python
from marl_battlegrounds import training
from marl_battlegrounds.baselines.config import {config_class}

config = training.TrainConfig(
    method="{entrant.method}",
    {config_field}={config_class}(input_scale={input_scale!r}),
    initial_actor=f"{{name}}@final",
)
```

The new run starts from this save's weights. The rest of the learner, such as
the critic, the mixer or the optimizer state, starts fresh. The method and input
scale must match this model's. Training state is not released, so the original
run cannot be continued; start a new run from any save instead.

### Choose a Save by Its Validation Result

```python
import csv

from huggingface_hub import hf_hub_download

path = hf_hub_download("{name}", "steps.csv")
with open(path, newline="") as file:
    rows = [row for row in csv.DictReader(file) if row["validation_games"]]
best = max(rows, key=lambda row: float(row["validation_point_margin"]))
system = marl_bgs.load_method(f"{{name}}@{{best['step']}}")
```

### Offline Use and Clusters

```bash
python -m marl_battlegrounds models download {name}@final
HF_HUB_OFFLINE=1 python my_experiment.py
```

The first command fills the Hugging Face cache. Later runs with
`HF_HUB_OFFLINE=1`, for example on compute nodes that share the cache, never use
the network.

### Download the Files Yourself

```bash
hf download {name} --local-dir {model.repo}
hf download {name} models/{model_file_name(near)} --local-dir {model.repo}
```

The first command fetches the whole repository (about {_size(repo_bytes)}), and
the second fetches one file. A downloaded folder loads with the same `@` names,
as in `marl_bgs.load_method("{model.repo}@100M")`, and a single file loads by its
path.

## What Is in This Repository

| Path | Holds |
|---|---|
| `models/step-XXX_XXX_XXX.zip` | One portable model file per save, named by its step. It holds one agent's decision function (exported with `jax.export`) and its weights, and runs under `jit` and `vmap` without the training code. |
| `steps.csv` | One row per file: `step`, `kind`, `file`, `bytes`, `sha256`, `original_checkpoint_id` (the save's id in the training records), `actor_digest`, the validation result where one exists, and `format`. |
| `config.json` | The method, the Season 0 save and the SHA-256 of `steps.csv`. |
| `training-records/{FIRST_RECORDS}/` | Records of the first training run, steps {first:,} to {boundary:,}: settings, learning curves, update logs and validation games. |
| `training-records/{SECOND_RECORDS}/` | Records of the run that continued it, steps {boundary:,} to {last:,}. |

Both runs hold the save at step {boundary:,}; this repository releases the first
run's copy.

## Validation Results in `steps.csv`

Training checked some saves against a fixed panel: {_members_text(panel)}. \
Each check played {games} games on validation maps {_numbers_text(panel.maps)}. \
`validation_point_margin` is this model's points minus its opponent's per game,
and `validation_kills` and `validation_deaths` are its kills and deaths per game.
All three are averaged over maps and opponents with equal weight. An empty cell
means the save was not checked. Larger confirmation checks are in
`training-records/*/validation_results.json`.

## Reproducibility

{claim} {_SAME_INPUTS}

Files run on JAX up to 6 months newer than the JAX that made them, as
[JAX promises for exported programs](https://docs.jax.dev/en/latest/export/export.html#compatibility-guarantees);
each file's `model.json` names its JAX version. MARL-BGs pins the JAX its
released files were made with and, before moving to a JAX outside that window,
republishes the files in a newer format as new files.

Every file was re-released once, on {RE_RELEASE_DAY}, to remove local folder
paths from the saved descriptions; programs and weights stayed byte-identical.
Model files are never replaced again, and MARL-BGs 1.0.1 and later pin these
exact commits.

## Season 0 Results

Every Season 0 game, rating and original save folder is in the dataset
[{ORG}/{DATASET_REPO}]({DATASET_URL}).

{citation_section(*citation)}
## License and Contact

Apache-2.0. Questions: {CONTACT}.
"""  # noqa: E501


def build_model_repo(
    model: Model,
    *,
    models_root: Path,
    out: Path,
    artifacts: Path,
    same_games: str,
    citation: tuple[str, str | None],
) -> Path:
    """Build one model repository folder and print what it holds.

    Parameters
    ----------
    model : Model
        The model to build.
    models_root : Path
        Holds ``<repo>/step-XXX_XXX_XXX.zip`` converted files.
    out : Path
        Parent folder. ``out/<repo>`` must not exist yet.
    artifacts : Path
        The project's ``artifacts/`` folder. It is only read.
    same_games : {"identical", "slightly-different"}
        The model card's "same games" claim.
    citation : tuple of (str, str or None)
        The paper's BibTeX entry and arXiv id for the card (see
        paper_citation).

    Returns
    -------
    Path
        The new ``out/<repo>`` folder.

    Raises
    ------
    PublishError
        Any check fails. All checks run before anything is written.

    Notes
    -----
    Model files are hard links when ``models_root`` and ``out`` share a file
    system, else copies. Record files are always copies, so nothing under
    ``artifacts/`` shares a file with the output.
    """
    search = artifacts / _SEARCH
    first, second = search / model.first_run, search / model.second_run
    saves = read_saves(first, second)
    entrant = read_entrant(artifacts / _RATINGS, model.case)
    entered = _entered_save(saves, entrant)
    files = match_model_files(models_root / model.repo, saves, entrant.method)
    validation = {**read_validation(first), **read_validation(second)}
    panel = read_panel(first, second)
    input_scale = _number(
        _read_json(entered.folder / "actor_details.json").get("input_scale")
    )
    if input_scale is None:
        raise PublishError(f"{entered.folder} has no input_scale")
    target = out / model.repo
    if target.exists():
        raise PublishError(f"{target} already exists; choose a new --out folder")
    (target / "models").mkdir(parents=True)
    links = sum(
        _link_or_copy(item.path, target / "models" / item.path.name) for item in files
    )
    records = [
        (f"training-records/{label}/{path}", size)
        for label, run in ((FIRST_RECORDS, first), (SECOND_RECORDS, second))
        for path, size in copy_records(run, target / "training-records" / label)
    ]
    rows = steps_rows(files, validation)
    _write_csv(target / "steps.csv", STEPS_COLUMNS, rows)
    config = {
        "library": "marl-battlegrounds",
        "game_mode": "tdm",
        "method": entrant.method,
        "display_name": entrant.label,
        "entered_step": entrant.step,
        "format": FILE_FORMAT,
        "steps_csv_sha256": _sha256(target / "steps.csv"),
        "season": SEASON,
    }
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model_bytes = sum(item.size for item in files)
    record_bytes = sum(size for _, size in records)
    small = (target / "steps.csv").stat().st_size + (
        target / "config.json"
    ).stat().st_size
    card = model_card(
        model,
        entrant,
        saves,
        panel,
        same_games,
        input_scale,
        model_bytes + record_bytes + small,
        citation,
    )
    (target / "README.md").write_text(card)
    print(
        f"{model.repo}: {len(files)} model files ({_size(model_bytes)}; {links} hard "
        f"links, {len(files) - links} copies), {len(records)} record files "
        f"({_size(record_bytes)}), steps.csv {len(rows)} rows "
        f"({sum(1 for row in rows if row[7] != '')} with validation)"
    )
    for path, size in records:
        if size > _LARGE_FILE:
            print(f"  large file: {path} ({_size(size)})")
    print(f"  wrote {target}")
    return target


def _copy_opponents(source: Path, target: Path) -> None:
    """Copy the training opponents bundle from a folder or a ZIP file.

    Parameters
    ----------
    source : Path
        A bundle folder (holding ``pool.json``), or a ZIP of one. A ZIP whose
        files all sit in one top folder is unpacked without that folder.
    target : Path
        New folder that receives ``pool.json`` and the model folders.

    Raises
    ------
    PublishError
        The source is neither a folder nor a ZIP, a ZIP entry would land
        outside ``target``, or the result has no ``pool.json``.
    """
    if source.is_dir():
        shutil.copytree(source, target)
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            parts = [PurePosixPath(entry.filename).parts for entry in entries]
            tops = {part[0] for part in parts}
            strip = len(tops) == 1 and all(len(part) > 1 for part in parts)
            for entry, part in zip(entries, parts, strict=True):
                inner = part[1:] if strip else part
                if (
                    not inner
                    or PurePosixPath(entry.filename).is_absolute()
                    or ".." in inner
                ):
                    raise PublishError(f"Unsafe path {entry.filename!r} in {source}")
                path = target.joinpath(*inner)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(entry))
    else:
        raise PublishError(f"{source} is neither a folder nor a ZIP file")
    if not (target / "pool.json").is_file():
        raise PublishError(f"{source} has no pool.json")


def _folder_bytes(folder: Path) -> int:
    """Return the total size of every file under ``folder``."""
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


# Small tables the dataset viewer shows: (name, path). full_metrics.csv has
# 11,222 columns, more than the viewer shows, so it stays a plain CSV file.
_VIEWER_TABLES = (
    ("ratings", "season-0-ratings/ratings.csv"),
    ("head_to_head", "season-0-ratings/head_to_head.csv"),
    ("map_results", "season-0-ratings/map_results.csv"),
    ("final_policies", "season-0-ratings/final_policies.csv"),
    ("entrants", "entrants.csv"),
    ("games", "season-0-games/match_results.csv"),
    ("matchups", "season-0-games/matchup_results.csv"),
    ("headline_metrics", "season-0-games/tournament_headline_metrics.csv"),
)


def dataset_card(
    games: int,
    maps: Sequence[int],
    panel_models: Sequence[str],
    citation: tuple[str, str | None],
) -> str:
    """Return the results dataset card (``README.md``).

    Parameters
    ----------
    games : int
        Number of games in the Season 0 tournament.
    maps : sequence of int
        The tournament's map ids, ascending.
    panel_models : sequence of str
        Folder names of the saved models in the model repositories' validation
        panels; each is under ``training-opponents/models/``.
    citation : tuple of (str, str or None)
        The paper's BibTeX entry and arXiv id (see paper_citation).

    Returns
    -------
    str
        Hugging Face YAML metadata, with ``configs`` naming only the small
        tables in ``_VIEWER_TABLES``, then the card in plain English.
    """
    metadata = DatasetCardData(
        license="apache-2.0",
        pretty_name="MARL-BattleGrounds Team Deathmatch Season 0 Results",
        tags=[
            "reinforcement-learning",
            "multi-agent-reinforcement-learning",
            "marl-battlegrounds",
            "team-deathmatch",
            *([f"arxiv:{citation[1]}"] if citation[1] else []),
        ],
        configs=[
            {
                "config_name": name,
                "data_files": path,
                **({"default": True} if index == 0 else {}),
            }
            for index, (name, path) in enumerate(_VIEWER_TABLES)
        ],
    ).to_yaml()
    panel = " and ".join(f"`{name}`" for name in panel_models)
    models = ", ".join(
        f"[{repo}](https://huggingface.co/{ORG}/{repo})" for repo in MODEL_REPOS
    )
    return f"""---
{metadata}
---

# MARL-BattleGrounds Team Deathmatch Season 0 Results

Season 0 of Team Deathmatch in [MARL-BattleGrounds]({CODE_URL}) (MARL-BGs) was
a round robin of nine teams: six trained models and the three scripted bots
TDM-Alpha, TDM-Beta and TDM-Gamma, which ship inside the package. They played
{games:,} games on test maps {_numbers_text(maps)}. This dataset holds every game,
the ratings, the original save folders and the training opponents.

Each model comes from a training run of about 250 million game steps.

| Path | Holds |
|---|---|
| `season-0-ratings/` | The Season 0 tables: `ratings.csv` (rank, Elo with its 95% interval, tier, wins, draws, losses, point margin, kills and deaths), `head_to_head.csv`, `map_results.csv`, `final_policies.csv` (each model's training settings) and `RESULTS.md`. |
| `season-0-games/` | A rerun of the Season 0 round robin that also saved every game's full metrics; every result matches the original run game for game. It holds `match_results.csv` (one row per game), per-pair and per-map tables, the tournament settings (`tournament_config.json`) and `full_metrics.csv`. That file has 11,222 columns, too wide for the dataset viewer, so it stays a plain CSV file. |
| `entrants.csv` | Each trained entrant's tournament name, released name, exact step and original checkpoint id. |
| `season-0-originals/<repo>/` | The original save folder that each model played Season 0 with. Its weights are unchanged. |
| `training-opponents/` | The training and validation opponents (`pool.json` and `models/`). The model repositories' validation results were measured against {panel} from here, with TDM-Alpha, TDM-Beta and TDM-Gamma. |

Each model has its own repository, with one portable file per save: {models}.

## Use It

```python
import csv

from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "{ORG}/{DATASET_REPO}", "season-0-ratings/ratings.csv", repo_type="dataset"
)
with open(path, newline="") as file:
    ratings = list(csv.DictReader(file))
print(ratings[0]["label"], ratings[0]["elo"])
```

```bash
hf download {ORG}/{DATASET_REPO} --repo-type dataset --local-dir {DATASET_REPO}
```

Every file was re-released once, on {RE_RELEASE_DAY}, to remove local folder
paths: saved descriptions and records now name models by their released names
and dataset addresses, and the weights are byte-identical.

{citation_section(*citation)}
## License and Contact

Apache-2.0. Questions: {CONTACT}.
"""  # noqa: E501


def build_dataset(
    *,
    out: Path,
    artifacts: Path,
    opponents: Path,
    citation: tuple[str, str | None],
) -> Path:
    """Build the results dataset folder and print what it holds.

    Parameters
    ----------
    out : Path
        Parent folder. ``out/tdm-season-0-results`` must not exist yet.
    artifacts : Path
        The project's ``artifacts/`` folder. It is only read.
    opponents : Path
        The training opponents bundle: a folder or a ZIP (see
        ``_copy_opponents``).
    citation : tuple of (str, str or None)
        The paper's BibTeX entry and arXiv id for the card (see
        paper_citation).

    Returns
    -------
    Path
        The new dataset folder.

    Raises
    ------
    PublishError
        A source is missing, a rated save is not among its run's saves, a
        validation panel model is not among the training opponents (the new
        folder is then removed again), or the target exists.

    Notes
    -----
    Everything is copied; nothing under ``artifacts/`` is changed or linked.
    """
    tournament, tables = artifacts / _TOURNAMENT, artifacts / _RATING_TABLES
    for source in (tournament, tables):
        if not source.is_dir():
            raise PublishError(f"Missing {source}")
    search = artifacts / _SEARCH
    entries: list[tuple[Model, Entrant, Save]] = []
    panel_models: set[str] = set()
    for model in MODELS:
        first, second = search / model.first_run, search / model.second_run
        saves = read_saves(first, second)
        entrant = read_entrant(artifacts / _RATINGS, model.case)
        entries.append((model, entrant, _entered_save(saves, entrant)))
        panel = read_panel(first, second)
        panel_models.update(folder for _, folder in panel.members if folder)
    target = out / DATASET_REPO
    if target.exists():
        raise PublishError(f"{target} already exists; choose a new --out folder")
    target.mkdir(parents=True)
    _copy_opponents(opponents, target / "training-opponents")
    absent = sorted(
        name
        for name in panel_models
        if not (target / "training-opponents/models" / name).is_dir()
    )
    if absent:
        shutil.rmtree(target)
        raise PublishError(f"Validation opponents missing from {opponents}: {absent}")
    shutil.copytree(tournament, target / "season-0-games")
    shutil.copytree(tables, target / "season-0-ratings")
    for model, _, save in entries:
        shutil.copytree(save.folder, target / "season-0-originals" / model.repo)
    _write_csv(
        target / "entrants.csv",
        ("entrant", "released_name", "step", "original_checkpoint_id"),
        (
            (
                entrant.entrant,
                f"{ORG}/{model.repo}@{step_text(save.step)}",
                save.step,
                save.checkpoint_id,
            )
            for model, entrant, save in entries
        ),
    )
    with (target / "season-0-games/match_results.csv").open(newline="") as file:
        games = sum(1 for _ in csv.DictReader(file))
    config = _read_json(target / "season-0-games/tournament_config.json")
    maps = sorted(
        int(_number(_object(item).get("map_id")) or 0)
        for item in _list(_object(config.get("conditions")).get("map_sources"))
    )
    missing = [path for _, path in _VIEWER_TABLES if not (target / path).is_file()]
    if missing:
        raise PublishError(f"Viewer tables missing: {missing}")
    (target / "README.md").write_text(
        dataset_card(games, maps, sorted(panel_models), citation)
    )
    for part in sorted(target.iterdir()):
        size = _folder_bytes(part) if part.is_dir() else part.stat().st_size
        print(f"{DATASET_REPO}/{part.name}: {_size(size)}")
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.stat().st_size > _LARGE_FILE:
            size = path.stat().st_size
            print(f"  large file: {path.relative_to(target)} ({_size(size)})")
    print(f"  wrote {target}")
    return target


@dataclass(frozen=True)
class OrgRow:
    """One model's row in the organization card's table.

    Attributes
    ----------
    repo : str
        Repository name inside the organization.
    label : str
        The model's display name, such as ``"Recurrent IPPO"``.
    rank : int
        Season 0 rank (1 is the top).
    elo : float
        Season 0 Elo.
    tier : str
        Season 0 tier, such as ``"Gold 3"``.
    saves : int
        Number of released saves (rows of ``steps.csv``).
    """

    repo: str
    label: str
    rank: int
    elo: float
    tier: str
    saves: int


def org_rows(release: Path) -> list[OrgRow]:
    """Read the organization card's table from a built release folder.

    Parameters
    ----------
    release : Path
        The ``--out`` folder of ``build`` and ``build-dataset``: it holds the six
        model repository folders and ``tdm-season-0-results/``. Only read.

    Returns
    -------
    list of OrgRow
        One row per model, in Season 0 rank order.

    Raises
    ------
    PublishError
        A repository folder, ``entrants.csv`` or ``ratings.csv`` is missing, or
        a model has no ratings row.
    """
    dataset = release / DATASET_REPO
    try:
        with (dataset / "entrants.csv").open(newline="") as file:
            entrants = {row["entrant"]: row for row in csv.DictReader(file)}
        with (dataset / "season-0-ratings/ratings.csv").open(newline="") as file:
            ratings = {row["entrant"]: row for row in csv.DictReader(file)}
        rows: list[OrgRow] = []
        for model in MODELS:
            with (release / model.repo / "steps.csv").open(newline="") as file:
                saves = sum(1 for _ in csv.DictReader(file))
            entrant = next(
                name
                for name, row in entrants.items()
                if row["released_name"].startswith(f"{ORG}/{model.repo}@")
            )
            rating = ratings[entrant]
            rows.append(
                OrgRow(
                    model.repo,
                    rating["label"],
                    int(rating["rank"]),
                    float(rating["elo"]),
                    rating["tier"],
                    saves,
                )
            )
    except (OSError, KeyError, StopIteration, ValueError) as error:
        raise PublishError(
            f"Cannot read the card table from {release}: {error}"
        ) from error
    return sorted(rows, key=lambda row: row.rank)


def org_card(rows: Sequence[OrgRow], citation: tuple[str, str | None]) -> str:
    """Return the organization card: the ``README.md`` of the Space ``README``.

    Parameters
    ----------
    rows : sequence of OrgRow
        The released models, in the order to show (see org_rows).
    citation : tuple of (str, str or None)
        The paper's BibTeX entry and arXiv id (see paper_citation).

    Returns
    -------
    str
        Space YAML metadata (static SDK), then the card in plain English: the
        tagline, the arXiv badge (when ``citation`` has an arXiv id), what
        MARL-BGs is, what wins and what its leaderboard rewards, the models
        with their Season 0 results, the one-line names, three lines of code,
        who may enter the leaderboard, links, the citation and the contact.
        Hugging Face shows this file as the organization card when the Space
        has no ``index.html``.
    """
    example = f"{ORG}/tdm-season-0-recurrent-ippo"
    arxiv = citation[1]
    badge = (
        f"[![arXiv](https://img.shields.io/badge/arXiv-{arxiv}-b31b1b.svg)]"
        f"(https://arxiv.org/abs/{arxiv})\n\n"
        if arxiv
        else ""
    )
    table = "\n".join(
        f"| [{row.label}](https://huggingface.co/{ORG}/{row.repo}) | {row.rank} | "
        f"{round(row.elo):,} | {row.tier} | {row.saves:,} |"
        for row in rows
    )
    return f"""---
title: MARL-BattleGrounds
sdk: static
pinned: false
---

# MARL-BattleGrounds

**The JAX-native Benchmark for Heterogeneous and Competitive Multi-agent Reinforcement Learning: built from scratch by researchers, for researchers.**

{badge}[MARL-BattleGrounds]({CODE_URL}) (MARL-BGs) is an easy-to-use, JAX-native
benchmark for heterogeneous and competitive team-vs-team multi-agent
reinforcement learning (MARL). It fills the missing middle: rich team play, with
fast and cheap experiments on one GPU. Its five classes are built to help allies
and counter enemies, so the team that best coordinates its specialists wins,
not the team with the best individual agents.

Every entry on its [leaderboard]({RULES_URL}) gets the same budget on the same
machine, so the leaderboard rewards methods that are efficient in both samples
and compute.

## The Season 0 Models

Each repository holds one trained Team Deathmatch (TDM) model: a portable JAX
file for every save up to 250 million steps, `steps.csv` (one row per save, with
its validation result where one exists) and the training records. Use them as
opponents, teammates or starting points for your own training.

| Model | Season 0 Rank | Elo | Tier | Saves |
|---|---|---|---|---|
{table}

The scripted bots TDM-Alpha, TDM-Beta and TDM-Gamma ship inside the package and
complete the Big Nine, the nine systems in the Season 0 table. Every game, rating
and original save folder is in the dataset [{ORG}/{DATASET_REPO}]({DATASET_URL}).

## Use One in Three Lines

```python
import marl_battlegrounds as marl_bgs

system = marl_bgs.load_method("{example}@final")
result = marl_bgs.evaluate(system, "tdm-alpha", num_episodes=20)
```

With no `@`, a name gives the save that played Season 0. Add `@final` (the last
save), `@100M` (the nearest save at or before 100,000,000 steps) or an exact step
such as `@99_942_400` to pick another. Each file downloads once, is checked
before use and needs no account or token. Saved results record the exact file
and commit, so a resumed run loads the same file.

## Enter the Leaderboard

Any method that runs as pure JAX may enter, not only MARL: learned, scripted,
search-based, hybrid or built on a language model. Each entry plays the current
Big Nine and takes the lowest member's place if its Elo is higher, so the
benchmark gets harder as entries improve. Every entrant and its
saved models are released here with its season's table, so the opponent pool
keeps growing. The [rules]({RULES_URL}) give each season's dates, and the
[README]({CODE_URL}#leaderboard) shows how to enter.

## Links

- Code: [{CODE_URL.removeprefix("https://")}]({CODE_URL})
- Install: `pip install marl-battlegrounds` ([PyPI](https://pypi.org/project/marl-battlegrounds/))
- How to use the released models: [the Released Models guide]({GUIDE_URL})
- Website: [marl-battlegrounds.com](https://marl-battlegrounds.com)
- Collections: **Season 0 Big Nine (TDM)** (the six models and the results) and
  **General Pool (TDM)** (every released entrant)

{citation_section(*citation)}
## Contact

Questions: {CONTACT}. Everything here is released under Apache-2.0.
"""  # noqa: E501


_SAVE_STEP = re.compile(r"_(?:actor|checkpoint|past)_step_(\d+)_")
# Where each kind of text file sits decides how a training opponent is named:
# files outside the dataset name the dataset with its commit; files inside it
# name the folder; the games folder also names the original save folders.
type Role = Literal["outside", "dataset", "games"]


@dataclass(frozen=True)
class PathMap:
    """Everything needed to turn this computer's paths into public names.

    Attributes
    ----------
    search : str
        Absolute path of ``artifacts/paper-experiments/hyperparameter-search``
        as it appears in the built files.
    runs : mapping of str to tuple of (str, str)
        Each training run folder's absolute path to its model repository and
        records label (``FIRST_RECORDS`` or ``SECOND_RECORDS``).
    released : mapping of str to str
        Each released save's absolute actor folder to its released name with
        its exact step, such as ``MARL-BattleGrounds/tdm-season-0-qmix@40_108_032``.
    originals : mapping of str to str
        Each Season 0 save's absolute actor folder to its model repository.
    dataset_commit : str or None
        The dataset commit that ``hf://`` addresses name. None while the
        dataset itself is being cleaned, before that commit exists.
    """

    search: str
    runs: Mapping[str, tuple[str, str]]
    released: Mapping[str, str]
    originals: Mapping[str, str]
    dataset_commit: str | None


def path_map(artifacts: Path, dataset_commit: str | None) -> PathMap:
    """Return the PathMap of the Season 0 release built from ``artifacts``.

    Parameters
    ----------
    artifacts : Path
        The ``artifacts/`` folder the release was built from. Only read.
    dataset_commit : str or None
        The dataset's squashed commit (40 hexadecimal characters), or None.

    Returns
    -------
    PathMap
        Built from the same runs, saves and ratings as ``build``.
    """
    root = Path(str(artifacts.resolve()))
    search = root / _SEARCH
    runs: dict[str, tuple[str, str]] = {}
    released: dict[str, str] = {}
    originals: dict[str, str] = {}
    for model in MODELS:
        first, second = search / model.first_run, search / model.second_run
        runs[str(first)] = (model.repo, FIRST_RECORDS)
        runs[str(second)] = (model.repo, SECOND_RECORDS)
        saves = read_saves(first, second)
        for step, save in saves.items():
            released[str(save.folder)] = f"{ORG}/{model.repo}@{step_text(step)}"
        entrant = read_entrant(root / _RATINGS, model.case)
        originals[str(_entered_save(saves, entrant).folder)] = model.repo
    return PathMap(str(search), runs, released, originals, dataset_commit)


def _dataset_address(paths: PathMap, folder: str) -> str:
    """Return the exact ``hf://`` address of a folder in the results dataset."""
    if paths.dataset_commit is None:
        raise PublishError(f"The dataset commit is needed to name {folder}")
    return f"hf://datasets/{ORG}/{DATASET_REPO}@{paths.dataset_commit}/{folder}"


def public_path(path: str, role: Role, paths: PathMap) -> str:
    """Return the public name of one local path from the release's build.

    Parameters
    ----------
    path : str
        An absolute path under ``paths.search``.
    role : {"outside", "dataset", "games"}
        Where the text sits: a model repository, the results dataset, or the
        dataset's games folder.
    paths : PathMap
        The release's path map.

    Returns
    -------
    str
        - A training opponent: ``training-opponents/models/<name>`` inside the
          dataset, else that folder's ``hf://`` dataset address.
        - A released save: its released name with its exact step; in the games
          folder, a Season 0 save names its original folder in the dataset.
        - A save, checkpoint or past copy that is not released:
          ``not published (<kind> at step N)``.
        - A run folder: ``training-records/<label>`` for role "outside" (a
          file of a model repository names only that model's own runs), else
          ``MARL-BattleGrounds/<repo>/training-records/<label>``.
        A path inside a folder keeps its remaining part after the new name.
        Any other path is returned unchanged, so the local path scan refuses it.
    """
    if not path.startswith(paths.search + "/"):
        return path
    rest = path[len(paths.search) + 1 :]
    if rest.startswith("opponents/models/"):
        folder = f"training-opponents/models/{rest.removeprefix('opponents/models/')}"
        return folder if role != "outside" else _dataset_address(paths, folder)
    for run, (repo, label) in paths.runs.items():
        if path != run and not path.startswith(run + "/"):
            continue
        records = (
            f"training-records/{label}"
            if role == "outside"
            else f"{ORG}/{repo}/training-records/{label}"
        )
        inside = path[len(run) + 1 :]
        head, _, below = inside.partition("/")
        if head not in ("actors", "checkpoints", "past_copies") or not below:
            return records if path == run else f"{records}/{inside}"
        folder, _, more = below.partition("/")
        full = f"{run}/{head}/{folder}"
        if role == "games" and full in paths.originals:
            name = _dataset_address(
                paths, f"season-0-originals/{paths.originals[full]}"
            )
        elif head == "actors" and full in paths.released:
            name = paths.released[full]
        else:
            found = _SAVE_STEP.search(folder)
            kind = {"actors": "save", "checkpoints": "checkpoint"}.get(
                head, "past copy"
            )
            step = f" at step {int(found.group(1)):,}" if found else ""
            name = f"not published ({kind}{step})"
        return f"{name}/{more}" if more else name
    return path


def public_text(text: str, role: Role, paths: PathMap) -> str:
    """Return a text file's content with every local name made public.

    Parameters
    ----------
    text : str
        The file's text.
    role : {"outside", "dataset", "games"}
        Where the file sits (see public_path).
    paths : PathMap
        The release's path map.

    Returns
    -------
    str
        The text with each path under ``paths.search`` replaced by public_path,
        each GPU id (``GPU-<uuid>``) replaced by ``GPU-<id>``, and the old
        repository address replaced by CODE_URL. Everything else is unchanged.
    """
    pattern = re.compile(re.escape(paths.search) + r"[^\s\"'`,;)\]}>|]*")
    text = pattern.sub(lambda match: public_path(match.group(0), role, paths), text)
    return _GPU_ID.sub("GPU-<id>", text).replace(OLD_CODE_URL, CODE_URL)


def _json_line(value: object) -> bytes:
    """Encode JSON as saved actor descriptions are: compact, sorted, one line."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def resign_description(raw: bytes, role: Role, paths: PathMap) -> bytes | None:
    """Return a saved actor description with public names and a new ID.

    Parameters
    ----------
    raw : bytes
        An ``actor_details.json``: compact sorted JSON whose ``checkpoint_id``
        is the SHA-256 of the description without that field.
    role : {"outside", "dataset", "games"}
        Where the description sits (see public_path).
    paths : PathMap
        The release's path map.

    Returns
    -------
    bytes or None
        None when nothing changes. Otherwise the cleaned description, with
        ``metadata.derived_from`` set to the old ``checkpoint_id`` and a new
        ``checkpoint_id`` computed by the same rule. The training run's own id,
        ``metadata.checkpoint_id``, never changes.

    Raises
    ------
    PublishError
        The old ``checkpoint_id`` does not match the description, so it was
        not a saved description.
    """
    details = _object(json.loads(raw))
    old = details.get("checkpoint_id")
    body = {key: value for key, value in details.items() if key != "checkpoint_id"}
    if old != hashlib.sha256(_json_line(body)).hexdigest() or raw != _json_line(
        details
    ):
        raise PublishError("A description's checkpoint_id does not match its content")
    cleaned = _object(json.loads(public_text(raw.decode(), role, paths)))
    if cleaned == details:
        return None
    cleaned.pop("checkpoint_id")
    cleaned["metadata"] = {**_object(cleaned.get("metadata")), "derived_from": old}
    cleaned["checkpoint_id"] = hashlib.sha256(_json_line(cleaned)).hexdigest()
    return _json_line(cleaned)


def clean_model_file(source: Path, target: Path, paths: PathMap) -> bool:
    """Write a portable model file whose description names no local path.

    Parameters
    ----------
    source : Path
        A format 1 portable model file. Only read.
    target : Path
        Where to write the result. It must not exist yet.
    paths : PathMap
        The release's path map, with its dataset commit.

    Returns
    -------
    bool
        True when the description changed; False when the file is copied
        byte for byte.

    Notes
    -----
    Only ``actor_details.json`` and ``model.json`` change: the new description
    and, in ``model.json``, its ``checkpoint_id`` and part hash. The file is
    written by the portable model module's own writer, so the program, weights
    and initial memory keep their exact bytes.
    """
    from marl_battlegrounds import portable_models

    with zipfile.ZipFile(source) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    new = resign_description(parts["actor_details.json"], "outside", paths)
    if new is None:
        shutil.copyfile(source, target)
        return False
    model = _object(json.loads(parts.pop("model.json")))
    model.pop("parts", None)
    model["checkpoint_id"] = json.loads(new)["checkpoint_id"]
    parts["actor_details.json"] = new
    portable_models._write(target, model, parts)  # pyright: ignore[reportPrivateUsage]
    return True


def _clean_games(source: Path, target: Path, paths: PathMap) -> None:
    """Write the games folder's settings and record with public names.

    ``tournament_config.json`` gets public controller references and its
    ``snapshot_id`` recomputed by the package's own rule;
    ``run_details.json`` gets the same references, the new settings, the new
    snapshot ID and the new SHA-256 of ``tournament_config.json``.
    """
    from marl_battlegrounds.evaluation.tournament_config import snapshot_identity

    old_config_bytes = (source / "tournament_config.json").read_bytes()
    config = _object(json.loads(public_text(old_config_bytes.decode(), "games", paths)))
    old_snapshot = str(config["snapshot_id"])
    config["snapshot_id"] = snapshot_identity(config)
    config_bytes = _json_line(config)
    (target / "tournament_config.json").write_bytes(config_bytes)
    text = public_text((source / "run_details.json").read_text(), "games", paths)
    text = text.replace(old_snapshot, str(config["snapshot_id"])).replace(
        hashlib.sha256(old_config_bytes).hexdigest(),
        hashlib.sha256(config_bytes).hexdigest(),
    )
    details = _object(json.loads(text))
    _object(details["details"])["canonical_config"] = config
    (target / "run_details.json").write_bytes(_json_line(details))


_GAMES_FILES = (
    "season-0-games/run_details.json",
    "season-0-games/tournament_config.json",
)


def clean_repo(source: Path, target: Path, paths: PathMap) -> int:
    """Write a copy of one built repository folder that names no local path.

    Parameters
    ----------
    source : Path
        A built model repository folder, or the built results dataset folder
        (its name decides which). Only read.
    target : Path
        A new folder for the copy. It must not exist yet.
    paths : PathMap
        The release's path map. A model repository needs its
        ``dataset_commit``; without one the dataset is cleaned without its two
        games files, which name that commit.

    Returns
    -------
    int
        The number of files whose content changed.

    Raises
    ------
    PublishError
        ``target`` exists, a description's ID does not match its content, or a
        model repository is cleaned without the dataset commit.

    Notes
    -----
    Text files get public_text. Saved descriptions (``actor_details.json``,
    also inside model files) are re-signed (see resign_description). A model
    repository's ``steps.csv`` gets each model file's new size and SHA-256, and
    ``config.json`` the new SHA-256 of ``steps.csv``. Other files are hard
    links when possible, else copies. Weights and programs never change. The
    card's "about N GB" line keeps the built folder's size; check it against
    the copy, since shorter paths can make the copy smaller.
    """
    if target.exists():
        raise PublishError(f"{target} already exists; choose a new --out folder")
    dataset = source.name == DATASET_REPO
    if not dataset and paths.dataset_commit is None:
        raise PublishError(f"{source.name} needs --dataset-commit")
    changed = 0
    files: dict[str, tuple[int, str]] = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        out = target / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        if relative in _GAMES_FILES or relative in ("steps.csv", "config.json"):
            continue
        role: Role = "dataset" if dataset else "outside"
        if path.suffix == ".zip" and relative.startswith("models/"):
            changed += clean_model_file(path, out, paths)
            files[relative] = (out.stat().st_size, _sha256(out))
            continue
        data = path.read_bytes()
        if path.name == "actor_details.json":
            new = resign_description(data, role, paths)
        elif path.suffix in (".json", ".jsonl", ".csv", ".md", ".yaml", ".txt"):
            text = public_text(data.decode(), role, paths)
            new = None if text == data.decode() else text.encode()
        else:
            new = None
        if new is None:
            _link_or_copy(path, out)
        else:
            out.write_bytes(new)
            changed += 1
    if dataset and paths.dataset_commit is not None:
        _clean_games(source / "season-0-games", target / "season-0-games", paths)
        changed += 2
    if not dataset:
        with (source / "steps.csv").open(newline="") as file:
            rows = list(csv.reader(file))
        header, body = rows[0], rows[1:]
        where = {name: header.index(name) for name in ("file", "bytes", "sha256")}
        for row in body:
            size, digest = files[row[where["file"]]]
            row[where["bytes"]], row[where["sha256"]] = str(size), digest
        _write_csv(target / "steps.csv", header, body)
        config = _read_json(source / "config.json")
        config["steps_csv_sha256"] = _sha256(target / "steps.csv")
        (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        changed += 2
    print(f"{source.name}: {changed} files changed, written to {target}")
    return changed


def _token_bytes() -> bytes | None:
    """Return the saved Hugging Face token's bytes, or None when there is none.

    Notes
    -----
    The value is only compared with file contents. It is never printed.
    """
    token = get_token()
    return token.encode() if token and len(token) >= 8 else None


def scan_bytes(data: bytes | mmap.mmap, token: bytes | None) -> set[str]:
    """Return the kinds of secret found in some bytes.

    Parameters
    ----------
    data : bytes or mmap.mmap
        The bytes to search.
    token : bytes or None
        The saved Hugging Face token to look for, or None to skip that check.

    Returns
    -------
    set of str
        Plain-English kinds, such as ``"GitHub token pattern"``. Empty when
        nothing matched. The matched text is never returned.
    """
    found = {kind for kind, pattern in _SECRETS.items() if pattern.search(data)}
    if token is not None and data.find(token) != -1:
        found.add(_SAVED_TOKEN)
    return found


def scan_folder(folder: Path) -> tuple[int, list[tuple[Path, list[str]]]]:
    """Search every file under ``folder`` for secret tokens and local paths.

    Parameters
    ----------
    folder : Path
        The folder to scan. Every regular file is read; ZIP files are also
        searched member by member, so compressed members are covered.

    Returns
    -------
    tuple of (int, list)
        The number of files scanned, and each file with findings and its
        sorted kinds. An unreadable ZIP counts as a finding. A local folder
        path or GPU id (see ``released_models._local_path_findings``) counts as
        ``"local path"``, with the first one found, because a path is not a
        secret and naming it shows where to fix it.
    """
    token = _token_bytes()
    findings: list[tuple[Path, list[str]]] = []
    paths = [path for path in sorted(folder.rglob("*")) if path.is_file()]
    for path in paths:
        found: set[str] = set()
        if path.stat().st_size:
            with (
                path.open("rb") as file,
                mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data,
            ):
                found = scan_bytes(data, token)
        if path.suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for entry in archive.infolist():
                        found |= scan_bytes(archive.read(entry), token)
            except (OSError, zipfile.BadZipFile):
                found.add("unreadable ZIP file")
        local = _local_path_findings(path.name, path.read_bytes())
        if local:
            found.add(f"local path ({len(local)} found, first: {local[0]})")
        if found:
            findings.append((path, sorted(found)))
    return len(paths), findings


def _report_scan(folder: Path) -> bool:
    """Scan a folder, print the result and return True when it is clean."""
    count, findings = scan_folder(folder)
    if not findings:
        print(f"scan: {count} files in {folder}, no secrets or local paths found")
        return True
    print(
        f"refused: {len(findings)} of {count} files in {folder} may hold secrets "
        "or local paths:"
    )
    for path, kinds in findings:
        print(f"  {path.relative_to(folder)}: {', '.join(kinds)}")
    return False


def _describe(name: str, args: Sequence[object], kwargs: Mapping[str, object]) -> str:
    """Return a call as text. Bytes show only their length."""

    def text(value: object) -> str:
        """Return one argument as text."""
        return f"<{len(value)} bytes>" if isinstance(value, bytes) else repr(value)

    parts = [text(value) for value in args]
    parts += [f"{key}={text(value)}" for key, value in kwargs.items()]
    return f"{name}({', '.join(parts)})"


def _call[**P, R](
    dry_run: bool, function: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> R | None:
    """Print one Hugging Face call, then make it unless this is a dry run.

    Parameters
    ----------
    dry_run : bool
        True prints ``would call ...`` and makes no call.
    function : callable
        A bound ``HfApi`` method.
    *args, **kwargs
        The call's arguments. Callers never pass a token.

    Returns
    -------
    object or None
        The call's result, or None in a dry run.
    """
    name = getattr(function, "__name__", "call")
    print(("would call " if dry_run else "call ") + _describe(name, args, kwargs))
    return None if dry_run else function(*args, **kwargs)


def remote_files(
    api: HfApi, repo_id: str, repo_type: str, revision: str | None
) -> dict[str, RepoFile]:
    """Return every file of a repository on the Hub, keyed by path.

    Parameters
    ----------
    api : HfApi
        The Hub client.
    repo_id, repo_type : str
        The repository.
    revision : str or None
        A commit, or None for the main branch.

    Returns
    -------
    dict of str to RepoFile
        Empty when the repository or revision does not exist yet.
    """
    try:
        return {
            entry.path: entry
            for entry in api.list_repo_tree(
                repo_id, recursive=True, revision=revision, repo_type=repo_type
            )
            if isinstance(entry, RepoFile)
        }
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError):
        return {}


def same_content(path: Path, remote: RepoFile, sha256: str) -> bool:
    """Return True when a Hub file holds exactly the local file's bytes.

    Parameters
    ----------
    path : Path
        The local file.
    remote : RepoFile
        The Hub's record of the file.
    sha256 : str
        The local file's SHA-256.

    Returns
    -------
    bool
        Large (LFS or Xet) files compare the Hub's SHA-256 and size; small
        files compare the Git blob id. Nothing is downloaded.
    """
    if remote.lfs is not None:
        return remote.lfs.sha256 == sha256 and remote.lfs.size == path.stat().st_size
    return remote.blob_id == _git_blob_id(path)


def upload(api: HfApi, folder: Path, *, private: bool, dry_run: bool) -> int:
    """Upload one built repository folder as a private repository and check it.

    Parameters
    ----------
    api : HfApi
        The Hub client.
    folder : Path
        A folder from ``build`` or ``build-dataset``; its name is the
        repository name.
    private : bool
        Must be True: new repositories are always private, and ``publish``
        makes them public later.
    dry_run : bool
        True prints the calls and makes none (the remote checks are skipped).

    Returns
    -------
    int
        0 when the upload is verified; 1 when the scan, the never-replace
        check or the verification refused.

    Raises
    ------
    PublishError
        The folder is not a known repository folder, or ``private`` is False.

    Notes
    -----
    Before uploading, every existing Hub file other than ``MUTABLE_FILES`` must
    already hold the local bytes, so released files are never replaced.
    Nothing is deleted. After uploading, every local file's SHA-256 (or Git
    blob id) must match the Hub's record at the new commit, whose id is printed.
    """
    if folder.name not in (*MODEL_REPOS, DATASET_REPO):
        raise PublishError(f"{folder.name} is not a release repository name")
    if not private:
        raise PublishError("upload creates private repositories only; pass --private")
    if not _report_scan(folder):
        return 1
    repo_id, repo_type = _repo(folder.name)
    local = {
        path.relative_to(folder).as_posix(): path
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }
    total = sum(path.stat().st_size for path in local.values())
    print(f"{repo_id}: {len(local)} local files, {_size(total)}")
    _call(
        dry_run,
        api.create_repo,
        repo_id,
        private=True,
        exist_ok=True,
        repo_type=repo_type,
    )
    hashes = {} if dry_run else {name: _sha256(p) for name, p in local.items()}
    if not dry_run:
        before = remote_files(api, repo_id, repo_type, None)
        print(f"{repo_id}: {len(before)} files already on the Hub")
        changed = [
            name
            for name in sorted(local)
            if name in before
            and name not in MUTABLE_FILES
            and not same_content(local[name], before[name], hashes[name])
        ]
        if changed:
            print(f"refused: {len(changed)} released files would change:")
            for name in changed:
                print(f"  {name}")
            return 1
    info = _call(
        dry_run,
        api.upload_folder,
        repo_id=repo_id,
        folder_path=folder,
        repo_type=repo_type,
        commit_message=f"Upload {folder.name}",
    )
    if dry_run:
        print("dry run: the remote checks before and after the upload were skipped")
        return 0
    if not isinstance(info, CommitInfo):
        raise PublishError("upload_folder did not return a commit")
    after = remote_files(api, repo_id, repo_type, info.oid)
    wrong = [
        name
        for name in sorted(local)
        if name not in after or not same_content(local[name], after[name], hashes[name])
    ]
    if wrong:
        print(f"refused: {len(wrong)} files differ on the Hub at {info.oid}:")
        for name in wrong:
            print(f"  {name}")
        return 1
    print(f"{repo_id}: verified {len(local)} files; commit {info.oid}")
    return 0


def _collection_slug(api: HfApi, title: str) -> str | None:
    """Return the slug of the organization's collection called ``title``."""
    for collection in api.list_collections(owner=ORG):
        if collection.title == title:
            return collection.slug
    return None


def collections(api: HfApi, *, private: bool, dry_run: bool) -> int:
    """Create or complete the two collections, as private.

    Parameters
    ----------
    api : HfApi
        The Hub client.
    private : bool
        Must be True; ``publish`` makes them public later.
    dry_run : bool
        True prints the calls and makes none.

    Returns
    -------
    int
        0 after every collection holds its items.

    Raises
    ------
    PublishError
        ``private`` is False.

    Notes
    -----
    Running it again reuses a collection with the same title and adds only the
    missing items, so it is safe to repeat. Each collection's URL is printed.
    """
    if not private:
        raise PublishError("collections are created private; pass --private")
    for title, description, items in COLLECTIONS:
        slug = None if dry_run else _collection_slug(api, title)
        if dry_run:
            print(f"dry run: would reuse a collection titled {title!r} if one exists")
        if slug is None:
            created = _call(
                dry_run,
                api.create_collection,
                title,
                namespace=ORG,
                description=description,
                private=True,
            )
            slug = created.slug if created is not None else f"{ORG}/<new>"
        for item_id, item_type in items:
            _call(
                dry_run,
                api.add_collection_item,
                slug,
                item_id,
                item_type,
                exists_ok=True,
            )
        print(f"{title}: https://huggingface.co/collections/{slug}")
    return 0


# The static Space's default page. With it, Hugging Face shows that page as the
# organization card; without it, Hugging Face shows README.md.
_SPACE_DEFAULTS = ("index.html", "style.css")


def _drop_space_defaults(api: HfApi, *, dry_run: bool) -> None:
    """Delete the Space's default index.html and style.css if they exist."""
    listed = _call(dry_run, api.list_repo_files, ORG_CARD_REPO, repo_type="space")
    present = set(_SPACE_DEFAULTS) if listed is None else set(listed)
    for name in _SPACE_DEFAULTS:
        if name in present:
            _call(
                dry_run,
                api.delete_file,
                name,
                repo_id=ORG_CARD_REPO,
                repo_type="space",
                commit_message=f"Remove the default {name}",
            )


def org_card_command(api: HfApi, release: Path, *, private: bool, dry_run: bool) -> int:
    """Upload the organization card, creating its Space when it does not exist.

    Parameters
    ----------
    api : HfApi
        The Hub client.
    release : Path
        The built release folder, read for the card's table (see org_rows).
    private : bool
        Visibility for a Space this call creates. An existing Space keeps its
        visibility.
    dry_run : bool
        True prints the calls and the card text and makes no call.

    Returns
    -------
    int
        0 after the upload; 1 when the card text fails the secret or local path
        scan.

    Raises
    ------
    PublishError
        The release folder lacks what the table needs, or README.md has no paper
        citation.

    Notes
    -----
    The Space's default ``index.html`` and ``style.css`` are deleted, so Hugging
    Face shows ``README.md`` as the organization card.
    """
    text = org_card(org_rows(release), paper_citation(_README.read_text())).encode()
    if scan_bytes(text, _token_bytes()) or _local_path_findings("README.md", text):
        print("refused: the organization card may hold a secret or a local path")
        return 1
    _call(
        dry_run,
        api.create_repo,
        ORG_CARD_REPO,
        repo_type="space",
        space_sdk="static",
        private=private,
        exist_ok=True,
    )
    _call(
        dry_run,
        api.upload_file,
        path_or_fileobj=text,
        path_in_repo="README.md",
        repo_id=ORG_CARD_REPO,
        repo_type="space",
        commit_message="Update the organization card",
    )
    _drop_space_defaults(api, dry_run=dry_run)
    if dry_run:
        print(text.decode())
    return 0


def cards(api: HfApi, *, dry_run: bool) -> int:
    """Bring every card's citation up to date with the README's.

    Run after README.md's ``## Citation`` changes, for example when the paper
    gets its arXiv number. Keep ``CITATION.cff``'s preferred citation the same;
    a test checks that the two agree.

    Parameters
    ----------
    api : HfApi
        The Hub client.
    dry_run : bool
        True prints the calls and makes none, not even downloads.

    Returns
    -------
    int
        0 after every card is current; 1 when a new card fails the scans.

    Raises
    ------
    PublishError
        README.md has no paper citation, or a card has no "Cite This Work"
        section.

    Notes
    -----
    Reads the six model cards, the dataset card and the organization card from
    Hugging Face, replaces only their "Cite This Work" sections and arXiv tags
    (see with_citation), and uploads only the cards that changed. Only
    ``README.md`` files change, so no model file or pinned commit changes. Needs
    nothing but this checkout's README.md and a token that can write to the
    organization.
    """
    citation = paper_citation(_README.read_text())
    targets = [
        *(_repo(name) for name in (*MODEL_REPOS, DATASET_REPO)),
        (ORG_CARD_REPO, "space"),
    ]
    for repo_id, repo_type in targets:
        if dry_run:
            print(f"dry run: would refresh the citation in {repo_id}")
            continue
        path = api.hf_hub_download(repo_id, "README.md", repo_type=repo_type)
        old = Path(path).read_text()
        new = with_citation(old, *citation)
        if new == old:
            print(f"{repo_id}: card already current")
            continue
        data = new.encode()
        if scan_bytes(data, _token_bytes()) or _local_path_findings("README.md", data):
            print(f"refused: the new card of {repo_id} may hold a secret or a path")
            return 1
        _call(
            dry_run,
            api.upload_file,
            path_or_fileobj=data,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Update the citation",
        )
        print(f"{repo_id}: card updated")
    _drop_space_defaults(api, dry_run=dry_run)
    return 0


def publish(api: HfApi, *, dry_run: bool) -> int:
    """Make the release public: repositories, collections, organization card.

    Parameters
    ----------
    api : HfApi
        The Hub client.
    dry_run : bool
        True prints the calls and makes none.

    Returns
    -------
    int
        0 after everything is public.

    Raises
    ------
    PublishError
        A collection does not exist yet (run ``collections`` first).
    """
    for name in (*MODEL_REPOS, DATASET_REPO):
        repo_id, repo_type = _repo(name)
        _call(
            dry_run,
            api.update_repo_settings,
            repo_id,
            private=False,
            repo_type=repo_type,
        )
    for title, _, _ in COLLECTIONS:
        slug = f"<{title}>" if dry_run else _collection_slug(api, title)
        if slug is None:
            raise PublishError(f"No collection titled {title!r}; run collections first")
        _call(dry_run, api.update_collection_metadata, slug, private=False)
    _call(
        dry_run,
        api.update_repo_settings,
        ORG_CARD_REPO,
        private=False,
        repo_type="space",
    )
    return 0


def redo(api: HfApi, name: str, *, dry_run: bool) -> int:
    """Delete and recreate one private ``tdm-season-0-*`` repository.

    Parameters
    ----------
    api : HfApi
        The Hub client.
    name : str
        The repository name, with or without the ``MARL-BattleGrounds/``
        prefix.
    dry_run : bool
        True prints the calls (including the privacy check) and makes none.

    Returns
    -------
    int
        0 after the repository is recreated empty and private; 1 when it is
        public, which is refused because released files are never deleted.

    Raises
    ------
    PublishError
        The name is not a ``tdm-season-0-*`` repository.
    """
    repo_id, repo_type = _repo(name)
    if not repo_id.removeprefix(f"{ORG}/").startswith("tdm-season-0-"):
        raise PublishError(
            f"redo works only on tdm-season-0-* repositories, not {name}"
        )
    info = _call(dry_run, api.repo_info, repo_id, repo_type=repo_type)
    if info is not None and info.private is not True:
        print(f"refused: {repo_id} is public; released files are never deleted")
        return 1
    _call(dry_run, api.delete_repo, repo_id, repo_type=repo_type)
    _call(dry_run, api.create_repo, repo_id, private=True, repo_type=repo_type)
    return 0


def main(argv: Sequence[str] | None = None, api: HfApi | None = None) -> int:
    """Run one command of the publishing tool.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        Command-line arguments without the program name. None reads
        ``sys.argv``.
    api : HfApi or None, default=None
        The Hub client for the Hugging Face commands. None makes a default
        ``HfApi``, which uses the saved login. Tests pass a fake.

    Returns
    -------
    int
        0 on success; 1 when a check refused (the reason is printed).
        Argument errors exit through argparse with status 2.
    """
    parser = argparse.ArgumentParser(
        description="Build, check and publish the released models on Hugging Face."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the model repository folders")
    build.add_argument(
        "--models-root",
        type=Path,
        required=True,
        help="folder with <repo>/step-XXX_XXX_XXX.zip files",
    )
    build.add_argument("--out", type=Path, required=True, help="output parent folder")
    build.add_argument(
        "--repo",
        nargs="+",
        choices=MODEL_REPOS,
        help="build only these repositories (default: all six)",
    )
    build.add_argument(
        "--same-games",
        required=True,
        choices=tuple(SAME_GAMES),
        help="the model cards' 'same games' claim",
    )
    dataset = commands.add_parser("build-dataset", help="build the results dataset")
    dataset.add_argument("--out", type=Path, required=True, help="output parent folder")
    dataset.add_argument(
        "--opponents-source",
        type=Path,
        default=None,
        help="training opponents bundle folder or ZIP "
        "(default: the website's training-opponents.zip)",
    )
    for command in (build, dataset):
        command.add_argument(
            "--artifacts",
            type=Path,
            default=_ARTIFACTS,
            help="the project's artifacts folder (read only)",
        )
    scan = commands.add_parser(
        "scan", help="refuse when any file holds a secret or a local path"
    )
    scan.add_argument("folder", type=Path)
    clean = commands.add_parser(
        "clean", help="copy built folders with local paths made public"
    )
    clean.add_argument("folders", type=Path, nargs="+", help="built repository folders")
    clean.add_argument("--out", type=Path, required=True, help="output parent folder")
    clean.add_argument(
        "--dataset-commit",
        default=None,
        help="the dataset's squashed commit; needed for model repositories and "
        "the dataset's games files",
    )
    clean.add_argument(
        "--artifacts",
        type=Path,
        default=_ARTIFACTS,
        help="the artifacts folder the release was built from (read only)",
    )
    send = commands.add_parser("upload", help="upload one built folder, private")
    send.add_argument("folder", type=Path)
    gather = commands.add_parser("collections", help="create the two collections")
    card = commands.add_parser("org-card", help="upload the organization card")
    card.add_argument(
        "--release", type=Path, required=True, help="the built release folder"
    )
    refresh = commands.add_parser(
        "cards", help="bring every card's citation up to date with README.md"
    )
    public = commands.add_parser("publish", help="make the whole release public")
    again = commands.add_parser("redo", help="delete and recreate a private repository")
    again.add_argument("repo")
    for command in (send, gather, card):
        command.add_argument(
            "--private",
            action="store_true",
            help="create private repositories (required for upload and collections)",
        )
    for command in (send, gather, card, refresh, public, again):
        command.add_argument(
            "--dry-run", action="store_true", help="print the calls and make none"
        )
    arguments = parser.parse_args(argv)
    try:
        match arguments.command:
            case "build":
                chosen = set(arguments.repo or MODEL_REPOS)
                citation = paper_citation(_README.read_text())
                for model in MODELS:
                    if model.repo in chosen:
                        build_model_repo(
                            model,
                            models_root=arguments.models_root,
                            out=arguments.out,
                            artifacts=arguments.artifacts,
                            same_games=arguments.same_games,
                            citation=citation,
                        )
                return 0
            case "build-dataset":
                source = (
                    arguments.opponents_source or arguments.artifacts / _OPPONENTS_ZIP
                )
                build_dataset(
                    out=arguments.out,
                    artifacts=arguments.artifacts,
                    opponents=source,
                    citation=paper_citation(_README.read_text()),
                )
                return 0
            case "scan":
                return 0 if _report_scan(arguments.folder) else 1
            case "clean":
                paths = path_map(arguments.artifacts, arguments.dataset_commit)
                for folder in arguments.folders:
                    target = arguments.out / folder.name
                    clean_repo(folder, target, paths)
                    if not _report_scan(target):
                        return 1
                return 0
            case _:
                pass
        hub = api if api is not None else HfApi()
        match arguments.command:
            case "upload":
                return upload(
                    hub,
                    arguments.folder,
                    private=arguments.private,
                    dry_run=arguments.dry_run,
                )
            case "collections":
                return collections(
                    hub, private=arguments.private, dry_run=arguments.dry_run
                )
            case "org-card":
                return org_card_command(
                    hub,
                    arguments.release,
                    private=arguments.private,
                    dry_run=arguments.dry_run,
                )
            case "cards":
                return cards(hub, dry_run=arguments.dry_run)
            case "publish":
                return publish(hub, dry_run=arguments.dry_run)
            case _:
                return redo(hub, arguments.repo, dry_run=arguments.dry_run)
    except PublishError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
