"""Check the release publishing tool on small fake inputs, with no network.

The tests build one fake model (two training runs, ratings and converted files)
and a fake Hugging Face client. They check these contracts of
``scripts/dev/publish_models.py``:

- ``steps.csv`` has one row per save in step order with the declared columns,
  the save both runs share at 40,108,032 comes from the first run, and the
  validation cells come from the routine check (empty for unchecked saves).
- ``config.json`` holds the season-neutral fields and the SHA-256 of
  ``steps.csv``; model files are hard links; training records leave out model
  folders, learner state, past copies and the run lock.
- The model card's metadata is read correctly by Hugging Face's own card
  parser, and the card states the chosen "same games" claim only.
- The secret scan refuses every token pattern, the saved token and a secret in
  a compressed ZIP member, and never prints the secret.
- ``upload`` checks every uploaded file's SHA-256 against the Hub's record and
  prints the commit; it refuses to change an existing model file on the Hub; ``redo``
  refuses a public repository; ``--dry-run`` makes no Hugging Face call.
"""

# The fake inputs reuse the tool's private artifact paths.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from huggingface_hub import HfApi, ModelCard, ModelCardData
from huggingface_hub.hf_api import CommitInfo, RepoFile
from scripts.dev import publish_models as publish

MODEL = next(model for model in publish.MODELS if model.repo == "tdm-season-0-qmix")
FIRST_STEPS = (0, 20_054_016, 40_108_032)
SECOND_STEPS = (40_108_032, 99_950_592, 219_095_040, 250_675_200)
ENTERED = 219_095_040


def _identifier(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _cell(
    opponent: str, points: tuple[float, float], kills: tuple[float, float]
) -> dict[str, object]:
    return {
        "opponent": opponent,
        "map_id": 42,
        "games": 8,
        "wins": 4,
        "draws": 0,
        "losses": 4,
        "mean_team_a_score": points[0],
        "mean_team_b_score": points[1],
        "mean_team_a_kills": kills[0],
        "mean_team_b_kills": kills[1],
    }


def _result(
    identifier: str, step: int, purpose: str, points_a: float
) -> dict[str, object]:
    return {
        "checkpoint_id": identifier,
        "env_steps": step,
        "purpose": purpose,
        "complete": True,
        "panel_digest": "panel",
        "maps": [42],
        "games": 16,
        "cells": [
            _cell("Recurrent MAPPO", (points_a, 3.0), (1.0, 2.0)),
            _cell("tdm-alpha", (2.0, 1.0), (2.0, 0.5)),
        ],
    }


def _run(
    root: Path, relative: str, steps: tuple[int, ...], checked: dict[int, float]
) -> dict[int, str]:
    run = root / publish._SEARCH / relative
    ids: dict[int, str] = {}
    for step in steps:
        identifier = _identifier(f"{relative}-{step}")
        ids[step] = identifier
        folder = run / "actors" / f"qmix_actor_step_{step:012d}_{identifier}"
        (folder / "actor").mkdir(parents=True)
        details = {
            "checkpoint_id": _identifier(f"export-{identifier}"),
            "actor_digest": _identifier(f"digest-{identifier}"),
            "input_scale": 0.01,
            "metadata": {"checkpoint_id": identifier, "env_steps": step},
        }
        (folder / "actor_details.json").write_text(json.dumps(details))
    results = [
        _result(ids[step], step, "routine", value) for step, value in checked.items()
    ]
    results.append(_result(ids[steps[-1]], steps[-1], "confirmation", 9.0))
    (run / "validation_results.json").write_text(json.dumps(results))
    (run / "run_details.json").write_text("{}")
    (run / ".training.lock").write_text("")
    for folder in ("checkpoints", "past_copies"):
        (run / folder).mkdir()
        (run / folder / "state.bin").write_bytes(b"state")
    return ids


@pytest.fixture(autouse=True)
def no_saved_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(publish, "get_token", lambda: None)


@pytest.fixture
def inputs(tmp_path: Path) -> SimpleNamespace:
    artifacts = tmp_path / "artifacts"
    first = _run(artifacts, MODEL.first_run, FIRST_STEPS, {0: 1.0, 40_108_032: 2.0})
    second = _run(artifacts, MODEL.second_run, SECOND_STEPS, {ENTERED: 4.0})
    panel = {
        "panel_digest": "panel",
        "members": [
            {
                "name": "Recurrent MAPPO",
                "reference": "/x/models/recurrent-mappo-pooled",
            },
            {"name": "tdm-alpha", "reference": "tdm-alpha"},
        ],
    }
    panel_file = (
        artifacts / publish._SEARCH / MODEL.first_run / "validation_panel/panel.json"
    )
    panel_file.parent.mkdir()
    panel_file.write_text(json.dumps(panel))
    entered = f"qmix_actor_step_{ENTERED:012d}_{second[ENTERED]}"
    ratings = artifacts / publish._RATINGS
    ratings.parent.mkdir(parents=True)
    ratings.write_text(
        "rank,entrant,label,method,case_id,seed,recipe,checkpoint_env_steps,"
        "reference,tier,elo,elo_ci_low,elo_ci_high\n"
        f'4,"QMIX at step 219,095,040 (run)",QMIX,qmix,{MODEL.case},12259871,'
        f"qmix-recipe,{ENTERED},/x/actors/{entered},Gold 3,1300.2,1287.6,1312.8\n"
        "7,tdm-beta,tdm-beta,scripted,,,,,tdm-beta,Silver 4,1042,1030,1055\n"
    )
    converted = tmp_path / "converted" / MODEL.repo
    converted.mkdir(parents=True)
    released = {
        **first,
        **{step: value for step, value in second.items() if step not in first},
    }
    for index, (step, identifier) in enumerate(sorted(released.items())):
        model = {
            "format": publish.PORTABLE_FORMAT,
            "format_version": 1,
            "method": "qmix",
            "display_name": "QMIX",
            "training_step": step,
            "checkpoint_id": identifier
            if index % 2
            else _identifier(f"export-{identifier}"),
            "actor_digest": _identifier(f"digest-{identifier}"),
        }
        with zipfile.ZipFile(converted / publish.model_file_name(step), "w") as archive:
            archive.writestr("model.json", json.dumps(model))
    return SimpleNamespace(
        artifacts=artifacts,
        converted=converted,
        first=first,
        second=second,
        out=tmp_path / "release",
    )


def _build(inputs: SimpleNamespace, same_games: str) -> Path:
    code = publish.main(
        [
            "build",
            "--models-root",
            str(inputs.converted.parent),
            "--out",
            str(inputs.out),
            "--repo",
            MODEL.repo,
            "--same-games",
            same_games,
            "--artifacts",
            str(inputs.artifacts),
        ]
    )
    assert code == 0
    return inputs.out / MODEL.repo


@pytest.mark.parametrize("same_games", ["identical", "slightly-different"])
def test_build_writes_index_config_records_and_card(
    inputs: SimpleNamespace, same_games: str
) -> None:
    target = _build(inputs, same_games)
    with (target / "steps.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == publish.STEPS_COLUMNS
    assert [int(row["step"]) for row in rows] == [
        0,
        20_054_016,
        40_108_032,
        99_950_592,
        ENTERED,
        250_675_200,
    ]
    shared = rows[2]
    assert shared["original_checkpoint_id"] == inputs.first[40_108_032]
    assert shared["original_checkpoint_id"] != inputs.second[40_108_032]
    # The routine check gives margin mean(2 - 3, 2 - 1) = 0.0; the confirmation
    # check at the same save (9.0 points) must not be used.
    assert [shared[key] for key in publish.STEPS_COLUMNS[7:11]] == [
        "16",
        "0.0",
        "1.5",
        "1.25",
    ]
    assert [rows[0][key] for key in publish.STEPS_COLUMNS[7:11]] == [
        "16",
        "-0.5",
        "1.5",
        "1.25",
    ]
    assert all(rows[1][key] == "" for key in publish.STEPS_COLUMNS[7:11])
    assert {row["kind"] for row in rows} == {"model"}
    assert {row["format"] for row in rows} == {"portable-model-1"}
    for row in rows:
        path = target / row["file"]
        source = inputs.converted / path.name
        assert os.path.samefile(path, source)
        assert row["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert int(row["bytes"]) == source.stat().st_size
    config = json.loads((target / "config.json").read_text())
    assert config == {
        "library": "marl-battlegrounds",
        "game_mode": "tdm",
        "method": "qmix",
        "display_name": "QMIX",
        "entered_step": ENTERED,
        "format": "portable-model-1",
        "steps_csv_sha256": hashlib.sha256(
            (target / "steps.csv").read_bytes()
        ).hexdigest(),
        "season": 0,
    }
    for label in ("0-40m", "40m-250m"):
        names = {path.name for path in (target / "training-records" / label).iterdir()}
        assert {"run_details.json", "validation_results.json"} <= names
        assert names.isdisjoint(
            {"actors", "checkpoints", "past_copies", ".training.lock"}
        )
    card = ModelCard.load(target / "README.md")
    data = cast(ModelCardData, card.data)
    assert data.license == "apache-2.0"
    assert data.library_name == "marl-battlegrounds"
    assert data.pipeline_tag == "reinforcement-learning"
    assert set(data.tags or ()) >= {
        "multi-agent-reinforcement-learning",
        "jax",
        "team-deathmatch",
    }
    assert [
        (result.metric_type, result.metric_value) for result in data.eval_results or ()
    ] == [("elo", 1300)]
    text = card.text
    other = next(choice for choice in publish.SAME_GAMES if choice != same_games)
    assert publish.SAME_GAMES[same_games] in text
    assert publish.SAME_GAMES[other] not in text
    assert "t.hawili@sms.ed.ac.uk" in text
    assert "@99_950_592" in text
    assert "recurrent-mappo-pooled" in text


class FakeHub:
    def __init__(
        self, remote: tuple[RepoFile, ...] = (), *, private: bool = True
    ) -> None:
        self.calls: list[str] = []
        self.remote = remote
        self.private = private

    def __getattr__(self, name: str) -> Callable[..., object]:
        def call(*args: object, **kwargs: object) -> object:
            self.calls.append(name)
            if name == "list_repo_tree":
                return iter(self.remote)
            if name == "list_collections":
                return iter(())
            if name == "repo_info":
                return SimpleNamespace(private=self.private)
            if name == "upload_folder":
                return CommitInfo(
                    commit_url=f"https://huggingface.co/{publish.ORG}/x/commit/c0ffee",
                    commit_message="",
                    commit_description="",
                    oid="c0ffee",
                )
            return SimpleNamespace(slug="MARL-BattleGrounds/pool-1")

        call.__name__ = name
        return call


def _remote(path: str, data: bytes, *, large: bool) -> RepoFile:
    if large:
        lfs = {
            "size": len(data),
            "oid": hashlib.sha256(data).hexdigest(),
            "pointerSize": 130,
        }
        return RepoFile(path=path, size=len(data), oid="0" * 40, lfs=lfs)
    blob = hashlib.sha1(
        b"blob %d\0" % len(data) + data, usedforsecurity=False
    ).hexdigest()
    return RepoFile(path=path, size=len(data), oid=blob)


@pytest.fixture
def repo_folder(tmp_path: Path) -> Path:
    folder = tmp_path / MODEL.repo
    (folder / "models").mkdir(parents=True)
    with zipfile.ZipFile(folder / "models" / "step-000_000_000.zip", "w") as archive:
        archive.writestr("model.json", "{}")
    (folder / "README.md").write_text("new card")
    return folder


def test_upload_refuses_to_replace_a_released_model_file(
    repo_folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = FakeHub(
        (
            _remote("models/step-000_000_000.zip", b"released model bytes", large=True),
            _remote("README.md", b"old card", large=False),
        )
    )
    assert (
        publish.main(["upload", str(repo_folder), "--private"], api=cast(HfApi, hub))
        == 1
    )
    assert "upload_folder" not in hub.calls
    output = capsys.readouterr().out
    assert "models/step-000_000_000.zip" in output
    assert "README.md" not in output.split("refused:")[1]


def test_upload_checks_every_file_after_uploading(
    repo_folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = repo_folder / "models" / "step-000_000_000.zip"
    hub = FakeHub(
        (
            _remote("models/step-000_000_000.zip", model.read_bytes(), large=True),
            _remote("README.md", b"new card", large=False),
        )
    )
    assert (
        publish.main(["upload", str(repo_folder), "--private"], api=cast(HfApi, hub))
        == 0
    )
    assert hub.calls == [
        "create_repo",
        "list_repo_tree",
        "upload_folder",
        "list_repo_tree",
    ]
    assert "verified 2 files; commit c0ffee" in capsys.readouterr().out


def test_redo_refuses_a_public_repository() -> None:
    hub = FakeHub(private=False)
    assert publish.main(["redo", MODEL.repo], api=cast(HfApi, hub)) == 1
    assert hub.calls == ["repo_info"]


@pytest.mark.parametrize(
    "command",
    [
        ["upload", "{folder}", "--private"],
        ["collections", "--private"],
        ["org-card", "--private"],
        ["publish"],
        ["redo", MODEL.repo],
    ],
)
def test_dry_run_makes_no_hugging_face_call(
    repo_folder: Path, command: list[str]
) -> None:
    hub = FakeHub()
    argv = [part.replace("{folder}", str(repo_folder)) for part in command]
    assert publish.main([*argv, "--dry-run"], api=cast(HfApi, hub)) == 0
    assert hub.calls == []


def _secrets() -> Iterator[tuple[str, bytes]]:
    yield "Hugging Face token pattern", b"hf_" + b"a1B2" * 9
    yield "GitHub token pattern", b"ghp_" + b"Z9y8" * 10
    yield "GitHub fine-grained token pattern", b"github_pat_" + b"x_Y7" * 6
    yield "PyPI token pattern", b"pypi-" + b"AgEI" + b"k-9_" * 5
    yield "AWS access key pattern", b"AKIA" + b"Q7W3E5R1T9Y2U4I6"
    yield "private key header", b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----"


@pytest.mark.parametrize("where", ["file", "zip member"])
@pytest.mark.parametrize(("kind", "secret"), list(_secrets()))
def test_scan_refuses_every_secret_without_printing_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    secret: bytes,
    where: str,
) -> None:
    if where == "file":
        (tmp_path / "notes.txt").write_bytes(b"before " + secret + b" after")
    else:
        with zipfile.ZipFile(
            tmp_path / "model.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("model.json", b"{}" + secret * 50)
    (tmp_path / "clean.txt").write_text("nothing to see")
    assert publish.main(["scan", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert kind in output
    assert "clean.txt" not in output
    assert secret.decode() not in output


def test_scan_refuses_the_saved_token_without_printing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    token = "saved-token-" + "q" * 20
    monkeypatch.setattr(publish, "get_token", lambda: token)
    (tmp_path / "log.txt").write_text(f"header {token} footer")
    assert publish.main(["scan", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "the saved Hugging Face token" in output
    assert token not in output
