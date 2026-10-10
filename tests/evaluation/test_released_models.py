"""Check released model names, their saved addresses and every place that takes them.

A fake Hugging Face (HF) stands in for the network: it serves steps.csv,
config.json and model files at fixed commits and writes them into a temporary
HF cache, so the real huggingface_hub cache lookup runs. No test reaches the
network. The portable model loader is faked; its file format is tested
elsewhere.

Contracts checked:

- Step rules: no suffix gives the entered save, @final the last save, @<N>M
  the nearest save at or before N million steps, and an exact step (with or
  without underscores) that save; a missing step lists the nearest saves.
  Owner and repo match without regard to case and are saved in HF's spelling.
- Suffixes that are not steps (a commit hash, a branch name) are refused, and
  TrainConfig checks typed names by spelling only, with no network call.
- A pinned repo whose files are cached loads with no network call. An
  unpinned repo looks up its latest commit once per process, warns, and its
  saved address keeps that commit even if main moves, so one study resolves
  each name once.
- A downloaded file whose SHA-256 differs from steps.csv is refused; offline
  with an uncached file fails at once and names the prefetch command.
- A local folder in the released layout takes the same @ rules and saves the
  chosen file's absolute path; an existing local path wins over a typed name.
- A typed name, its hf:// address and the same file on disk have one content
  identity, so a run can move between them.
- Resume resolves typed names against the commits the saved run recorded.
- The call sites (tournament inputs, studies, the trainer's config and
  population, the setup cache key, collection evidence, saved-reference and
  tournament descriptor checks) all use the one shared helper.
- download_models and ``models download`` fetch named files only.
- initial_actor from a released save gives the same System as the original
  actor folder.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import huggingface_hub
import huggingface_hub.constants
import huggingface_hub.errors
import jax
import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import _cli, portable_models, released_models
from marl_battlegrounds._method_loading import (
    _reference_identity,
    _reference_key,
    _reference_kind,
    load_method,
    validate_saved_method_reference,
)
from marl_battlegrounds.baselines.ppo import initialize_ppo
from marl_battlegrounds.evaluation import tournament_config
from marl_battlegrounds.evaluation._tournaments import tournament_inputs
from marl_battlegrounds.training import checkpoints, collection, runner, study
from marl_battlegrounds.training.config import TrainConfig

ORG = "MARL-BattleGrounds"
REPO = f"{ORG}/tdm-season-0-qmix"
COMMIT = "a" * 40
NEWER = "b" * 40
STEPS = (0, 40_108_032, 100_139_008, 249_856_000)
ENTERED = 100_139_008


def _file(step: int) -> str:
    return f"models/step-{step:011_}.zip"


class FakeHub:
    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.files: dict[tuple[str, str, str], bytes] = {}
        self.spelling: dict[str, str] = {}
        self.main: dict[str, str] = {}
        self.downloads: list[str] = []
        self.lookups: list[str] = []
        self.network = True

    def publish(self, repo: str, commit: str, *, corrupt: int | None = None) -> None:
        rows = io.StringIO()
        writer = csv.writer(rows)
        writer.writerow(
            [
                "step", "kind", "file", "bytes", "sha256", "original_checkpoint_id",
                "actor_digest", "validation_games", "validation_point_margin",
                "validation_kills", "validation_deaths", "format",
            ]
        )  # fmt: skip
        key = repo.lower()
        for step in STEPS:
            data = b"PORTABLE" + str(step).encode()
            digest = hashlib.sha256(data).hexdigest()
            row: list[object] = [step, "model", _file(step), len(data), digest]
            row.extend(["c" * 64, "d" * 64, "", "", "", "", "portable-model-1"])
            writer.writerow(row)
            self.files[key, commit, _file(step)] = (
                data + b"!" if step == corrupt else data
            )
        self.files[key, commit, "steps.csv"] = rows.getvalue().encode()
        self.files[key, commit, "config.json"] = json.dumps(
            {"entered_step": ENTERED}
        ).encode()
        self.spelling[key] = repo
        self.main[key] = commit

    def path(self, repo: str, commit: str, filename: str) -> Path:
        folder = f"models--{self.spelling[repo.lower()].replace('/', '--')}"
        return self.cache / folder / "snapshots" / commit / filename

    def download(self, repo_id: str, filename: str, **options: object) -> str:
        if not self.network:
            raise huggingface_hub.errors.LocalEntryNotFoundError("offline")
        self.downloads.append(filename)
        key = repo_id.lower(), str(options["revision"]), filename
        if key not in self.files:
            raise huggingface_hub.errors.EntryNotFoundError(filename)
        target = self.path(repo_id, key[1], filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[key])
        return str(target)

    def api(self, **_options: object) -> SimpleNamespace:
        def model_info(repo: str, revision: str = "main") -> SimpleNamespace:
            self.lookups.append(repo)
            if repo.lower() not in self.main:
                response: Any = SimpleNamespace(headers={}, request=None)
                raise huggingface_hub.errors.RepositoryNotFoundError(
                    "missing", response=response
                )
            return SimpleNamespace(
                id=self.spelling[repo.lower()], sha=self.main[repo.lower()]
            )

        return SimpleNamespace(model_info=model_info)


def _portable(path: str | Path) -> bool:
    return Path(path).read_bytes().startswith(b"PORTABLE")


def _load(path: str | Path) -> tuple[str, Path]:
    return "loaded", Path(path)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub(tmp_path / "hf-cache")
    fake.publish(REPO, COMMIT)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(fake.cache))
    monkeypatch.setattr(huggingface_hub.constants, "is_offline_mode", lambda: False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.download)
    monkeypatch.setattr(huggingface_hub, "HfApi", fake.api)
    monkeypatch.setattr(released_models, "PINNED_COMMITS", {REPO: COMMIT})
    monkeypatch.setattr(released_models, "_MAIN_COMMITS", {})
    monkeypatch.setattr(portable_models, "is_portable_model_file", _portable)
    monkeypatch.setattr(portable_models, "load_portable_model", _load)
    monkeypatch.chdir(tmp_path)
    return fake


def _address(step: int, commit: str = COMMIT, repo: str = REPO) -> str:
    return f"hf://{repo}@{commit}/{_file(step)}"


@pytest.mark.parametrize(
    ("suffix", "step"),
    [
        ("", ENTERED),
        ("@final", 249_856_000),
        ("@100M", 40_108_032),
        ("@101M", 100_139_008),
        ("@250M", 249_856_000),
        ("@100_139_008", 100_139_008),
        ("@100139008", 100_139_008),
    ],
)
def test_step_rules_choose_one_save_and_save_its_exact_address(
    hub: FakeHub, suffix: str, step: int
) -> None:
    assert released_models._saved_reference(REPO + suffix) == _address(step)
    shouted = "marl-battlegrounds/TDM-SEASON-0-QMIX" + suffix
    assert released_models._saved_reference(shouted) == _address(step)


@pytest.mark.parametrize(
    ("suffix", "message"),
    [
        ("@100_000_000", "nearest saves: 0, 40_108_032, 100_139_008, 249_856_000"),
        ("@1", "no save at exactly step 1; nearest saves: 0, 40_108_032"),
        ("@" + "c" * 40, "must choose a training step"),
        ("@main", "must choose a training step"),
    ],
)
def test_missing_steps_list_nearest_saves_and_other_suffixes_are_refused(
    hub: FakeHub, suffix: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_method(REPO + suffix)


def test_trainconfig_checks_typed_names_by_spelling_only(hub: FakeHub) -> None:
    hub.network = False
    config = TrainConfig(
        pinned_opponent=REPO + "@100M",
        pinned_opponent_share=0.2,
        initial_actor=REPO,
        validation_opponents=(REPO + "@final", "tdm-alpha"),
        slot_diagnostic_actor=REPO + "@final",
    )
    assert config.pinned_opponent == REPO + "@100M"
    with pytest.raises(ValueError, match="must choose a training step"):
        TrainConfig(pinned_opponent=REPO + "@main", pinned_opponent_share=0.2)
    assert hub.downloads == [] and hub.lookups == []


def test_cached_pinned_name_loads_with_no_network_call(
    hub: FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("steps.csv", "config.json", _file(ENTERED)):
        target = hub.path(REPO, COMMIT, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(hub.files[REPO.lower(), COMMIT, name])

    def no_network(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("A cached pinned model must not use the network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    monkeypatch.setattr(huggingface_hub, "HfApi", no_network)
    assert load_method(REPO) == ("loaded", hub.path(REPO, COMMIT, _file(ENTERED)))


def test_unpinned_repo_uses_latest_commit_once_and_keeps_it(
    hub: FakeHub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(released_models, "PINNED_COMMITS", {})
    declared = {"opponents": {REPO + "@final": 1.0}, "validation_opponents": [REPO]}
    with pytest.warns(UserWarning, match="not pinned"):
        first = study._config_paths(declared, tmp_path)
    hub.publish(REPO, NEWER)
    second = study._config_paths(declared, tmp_path)
    assert (
        first
        == second
        == {
            "opponents": {_address(249_856_000): 1.0},
            "validation_opponents": [_address(ENTERED)],
        }
    )
    assert hub.lookups == [REPO]
    with pytest.raises(ValueError, match="is not a local path, and Hugging Face"):
        released_models._saved_reference(f"{ORG}/missing")


def test_bad_hash_and_offline_misses_are_refused(
    hub: FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub.publish(REPO, NEWER, corrupt=ENTERED)
    monkeypatch.setattr(released_models, "PINNED_COMMITS", {REPO: NEWER})
    with pytest.raises(ValueError, match="does not match its SHA-256"):
        load_method(REPO)
    hub.network = False
    with pytest.raises(ValueError, match=f"models download {REPO}@final"):
        load_method(REPO + "@final")


def test_local_released_folder_and_identity_follow_the_file(hub: FakeHub) -> None:
    folder = Path("downloads") / "tdm-season-0-qmix"
    for name in ("steps.csv", "config.json", _file(0), _file(249_856_000)):
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(hub.files[REPO.lower(), COMMIT, name])
    final = str((folder / _file(249_856_000)).resolve())
    assert released_models._saved_reference(f"{folder}@final") == final
    assert load_method(f"./{folder}@0") == ("loaded", (folder / _file(0)).resolve())
    with pytest.raises(ValueError, match="is missing"):
        load_method(str(folder))
    identity = {"kind": "portable_model", "sha256": hashlib.sha256(
        hub.files[REPO.lower(), COMMIT, _file(249_856_000)]).hexdigest()}  # fmt: skip
    for form in (REPO + "@final", _address(249_856_000), f"{folder}@final", final):
        assert _reference_identity(form) == identity
    local = Path(REPO + "@final")
    local.mkdir(parents=True)
    assert _reference_kind(REPO + "@final") == "artifact"


def test_resume_uses_the_commits_the_saved_run_recorded(
    hub: FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub.publish(REPO, NEWER)
    monkeypatch.setattr(released_models, "PINNED_COMMITS", {REPO: NEWER})
    saved = {"collection": {"opponent_members": [{"reference": _address(0)}]}}
    commits = released_models._recorded_commits(saved)
    assert commits == {REPO: COMMIT}
    supplied = TrainConfig(pinned_opponent=REPO + "@final", pinned_opponent_share=0.2)
    resumed = runner._released_config(supplied, commits=commits)
    assert resumed.pinned_opponent == _address(249_856_000)
    assert runner._released_config(supplied).pinned_opponent == _address(
        249_856_000, NEWER
    )
    both = released_models._recorded_commits([_address(0), _address(0, NEWER)])
    with pytest.raises(ValueError, match="more than one commit"):
        released_models._saved_reference(REPO, commits=both)


def test_every_saving_place_uses_the_one_shared_helper(
    hub: FakeHub, tmp_path: Path
) -> None:
    name, address = REPO + "@101M", _address(100_139_008)
    config = runner._released_config(
        TrainConfig(
            pinned_opponent=name,
            pinned_opponent_share=0.2,
            validation_opponents=(name,),
            validation_partners={"mate": name},
        )
    )
    population, references = runner._population_bindings({name: 1.0}, None)
    assert population == {name: ("loaded", hub.path(REPO, COMMIT, _file(100_139_008)))}
    assert {
        tournament_inputs._reference(name, tmp_path),
        study._reference(name, tmp_path),
        config.pinned_opponent,
        *(config.validation_opponents or ()),
        *(config.validation_partners or {}).values(),
        references[name],
        _reference_key(name),
    } == {address}
    assert collection._export_path(address) == hub.path(
        REPO, COMMIT, _file(100_139_008)
    )
    validate_saved_method_reference(address)
    validate_saved_method_reference(f"hf://datasets/{ORG}/results@{COMMIT}/folder")
    content = dict.fromkeys(tournament_config._CONTENT_FIELDS)
    controller = {"kind": "reference", "reference": address, "content": content}
    tournament_config._controller(controller, {}, version=2)
    with pytest.raises(ValueError, match="full 40-character commit"):
        validate_saved_method_reference(f"hf://{REPO}@main/{_file(0)}")


def test_download_models_and_command_fetch_named_files_only(
    hub: FakeHub, capsys: pytest.CaptureFixture[str]
) -> None:
    assert marl_bgs.download_models() == []
    assert hub.downloads == []
    assert marl_bgs.download_models(REPO + "@final", REPO) == [
        hub.path(REPO, COMMIT, _file(249_856_000)),
        hub.path(REPO, COMMIT, _file(ENTERED)),
    ]
    assert _cli.main(["models", "download", REPO + "@0"]) == 0
    assert f"{REPO}@0: {_address(0)}" in capsys.readouterr().out


def test_initial_actor_from_a_released_save_equals_the_original_folder(
    hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "actors").mkdir()
    folder = checkpoints.export_system(
        initialize_ppo(jax.random.key(3), method="mappo").actor_params,
        tmp_path / "actors" / "actor",
        metadata={"run_id": "r", "seed": 3, "env_steps": 0, "checkpoint_id": "b" * 64},
        spawn_frame="left",
        method="mappo",
    )
    details = checkpoints.read_checkpoint_description(folder)
    template = checkpoints._checked_actor_template(details)
    restored = checkpoints._restore_arrays(folder / "actor", template, None)
    weights = jax.tree.map(np.asarray, restored)
    fetched: list[Path] = []

    def actor_variables(path: str | Path) -> tuple[Any, dict[str, Any]]:
        fetched.append(Path(path))
        return weights, json.loads(json.dumps(details))

    monkeypatch.setattr(portable_models, "portable_actor_variables", actor_variables)
    options: dict[str, Any] = {
        "method": "mappo", "input_scale": 1.0, "spawn_frame": "left"
    }  # fmt: skip
    original, _ = checkpoints.load_initial_actor(folder, **options)
    released, released_details = checkpoints.load_initial_actor(REPO, **options)
    assert fetched == [hub.path(REPO, COMMIT, _file(ENTERED))]
    assert released_details == details
    assert released.checkpoint == original.checkpoint
    assert released.name == original.name
    assert jax.tree.all(
        jax.tree.map(np.array_equal, released.variables, original.variables)
    )
