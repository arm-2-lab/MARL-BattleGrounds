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
- The shared local path rule (``released_models._local_path_findings``) finds
  home folders, Windows user folders and GPU ids, also inside a model file's
  JSON, in PATH lists, after ``host:``, in WSL's ``/mnt/c/Users`` and in
  ``file://`` URLs, and skips URLs, ``hf://`` addresses, author names and
  binary data.
- ``public_text`` gives each kind of local path its public name: training
  opponents (dataset address outside the dataset, folder inside it), released
  saves, unreleased saves, checkpoints and past copies, run folders, the Season 0
  originals in the games folder, GPU ids and the old repository address.
- A re-signed description keeps the ID rule, names its old ID in
  ``metadata.derived_from`` and keeps the run's own ID; a cleaned model file
  keeps its program, weights and memory bytes.
- Cards cite the paper from the README's citation, never the software;
  ``with_citation`` replaces only that section and adds the arXiv tag. The
  real README's paper citation names the same title and authors, in order, as
  ``CITATION.cff``'s preferred citation.
"""

# The fake inputs reuse the tool's private artifact paths.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
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


PAPER = """@article{hawili2026marlbattlegrounds,
  title   = {{MARL-BattleGrounds}: A Benchmark},
  author  = {Hawili, Ulixes Tariq and Ramamoorthy, Subramanian},
  journal = {arXiv preprint},
  year    = {2026}
}"""


@pytest.fixture(autouse=True)
def no_saved_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(publish, "get_token", lambda: None)


@pytest.fixture(autouse=True)
def paper_readme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# X\n\n## Citation\n\nCite:\n\n```bibtex\n{PAPER}\n```\n\n## License\n"
    )
    monkeypatch.setattr(publish, "_README", readme)
    return readme


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
    assert PAPER in text and "@software" not in text
    assert publish.GUIDE_URL in text


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
        ["cards"],
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


def test_local_path_rule_finds_paths_and_skips_names_urls_and_binary() -> None:
    from marl_battlegrounds.released_models import _local_path_findings

    assert _local_path_findings("a.json", b'{"r": "/home/sam/runs/actors/a"}') == [
        "a folder on this computer: /home/sam/runs/actors/a"
    ]
    clean = (
        b"Hawili, Ulixes Tariq; https://github.com/arm-2-lab/x; hf://datasets/o/r@c/x"
    )
    assert _local_path_findings("card.md", clean) == []
    table = b"ref\n/Users/sam/y\nC:\\Users\\sam\\x\n"
    assert _local_path_findings("t.csv", table) == [
        "a folder on this computer: /Users/sam/y",
        "a folder on this computer: C:\\Users\\sam\\x",
    ]
    hidden = (
        b"PATH=/usr/bin:/home/sam/bin\nhost:/home/kim/a\nC:/Users/sam/w\n"
        b"/mnt/c/Users/sam/v\nfile:///home/lee/u\n"
    )
    assert _local_path_findings("e.txt", hidden) == [
        "a folder on this computer: /home/sam/bin",
        "a folder on this computer: /home/kim/a",
        "a folder on this computer: C:/Users/sam/w",
        "a folder on this computer: /mnt/c/Users/sam/v",
        "a folder on this computer: /home/lee/u",
    ]
    gpu = b'{"d": "GPU-6b11a0c4-14e8-6782-8932-1df56d599796"}'
    assert _local_path_findings("d.json", gpu) == [
        "a GPU id: GPU-6b11a0c4-14e8-6782-8932-1df56d599796"
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("actor_details.json", '{"r": "/home/sam/opp"}')
        archive.writestr("weights.npz", b"/home/zzz/" * 3)
    assert _local_path_findings("m.zip", buffer.getvalue()) == [
        "a folder on this computer: /home/sam/opp"
    ]
    assert _local_path_findings("w.npz", b"\x00/home/abc/def") == []


SEARCH = "/home/sam/artifacts/paper-experiments/hyperparameter-search"
RUN = f"{SEARCH}/250m/study/cases/qmix--seed-1/run"
SAVE = f"{RUN}/actors/qmix_actor_step_000000000100_{'a' * 64}"
ORIGINAL = f"{RUN}/actors/qmix_actor_step_000000000200_{'b' * 64}"
PATHS = publish.PathMap(
    search=SEARCH,
    runs={RUN: (MODEL.repo, "40m-250m")},
    released={SAVE: f"{publish.ORG}/{MODEL.repo}@100", ORIGINAL: "x@200"},
    originals={ORIGINAL: MODEL.repo},
    dataset_commit="c" * 40,
)


@pytest.mark.parametrize(
    ("path", "role", "public"),
    [
        (
            f"{SEARCH}/opponents/models/opp-step-5",
            "outside",
            f"hf://datasets/{publish.ORG}/{publish.DATASET_REPO}@{'c' * 40}"
            "/training-opponents/models/opp-step-5",
        ),
        (
            f"{SEARCH}/opponents/models/opp-step-5",
            "dataset",
            "training-opponents/models/opp-step-5",
        ),
        (SAVE, "outside", f"{publish.ORG}/{MODEL.repo}@100"),
        (
            f"{SAVE}/actor/_METADATA",
            "outside",
            f"{publish.ORG}/{MODEL.repo}@100/actor/_METADATA",
        ),
        (
            f"{RUN}/actors/qmix_actor_step_000000000300_{'d' * 64}",
            "outside",
            "not published (save at step 300)",
        ),
        (
            f"{RUN}/checkpoints/qmix_checkpoint_step_000040108032_{'e' * 64}/x.json",
            "outside",
            "not published (checkpoint at step 40,108,032)/x.json",
        ),
        (
            f"{RUN}/past_copies/qmix_past_step_1376256_capture_000000",
            "outside",
            "not published (past copy at step 1,376,256)",
        ),
        (RUN, "outside", "training-records/40m-250m"),
        (
            f"{RUN}/validation/panel.json",
            "dataset",
            f"{publish.ORG}/{MODEL.repo}/training-records/40m-250m/validation/panel.json",
        ),
        (
            ORIGINAL,
            "games",
            f"hf://datasets/{publish.ORG}/{publish.DATASET_REPO}@{'c' * 40}"
            f"/season-0-originals/{MODEL.repo}",
        ),
        (ORIGINAL, "dataset", "x@200"),
        ("/home/sam/elsewhere/x", "outside", "/home/sam/elsewhere/x"),
    ],
)
def test_public_path_names_every_kind_of_local_path(
    path: str, role: publish.Role, public: str
) -> None:
    assert publish.public_path(path, role, PATHS) == public


def test_public_text_replaces_paths_gpu_ids_and_the_old_address() -> None:
    text = (
        f'{{"a": "{SAVE}", "gpu": "GPU-6b11a0c4-14e8-6782-8932-1df56d599796", '
        f'"code": "{publish.OLD_CODE_URL}/blob/main/x"}}\n`{RUN}`.'
    )
    assert publish.public_text(text, "outside", PATHS) == (
        f'{{"a": "{publish.ORG}/{MODEL.repo}@100", "gpu": "GPU-<id>", '
        f'"code": "{publish.CODE_URL}/blob/main/x"}}\n`training-records/40m-250m`.'
    )


def _description(reference: str) -> bytes:
    body = {
        "actor_digest": "d",
        "metadata": {
            "checkpoint_id": "run-id",
            "opponent_members": [{"reference": reference}],
        },
    }
    details = {
        **body,
        "checkpoint_id": hashlib.sha256(publish._json_line(body)).hexdigest(),
    }
    return publish._json_line(details)


def test_resign_description_keeps_the_id_rule_and_names_the_old_id() -> None:
    raw = _description(f"{SEARCH}/opponents/models/opp")
    old = json.loads(raw)
    new = json.loads(cast(bytes, publish.resign_description(raw, "dataset", PATHS)))
    body = {key: value for key, value in new.items() if key != "checkpoint_id"}
    assert new["checkpoint_id"] == hashlib.sha256(publish._json_line(body)).hexdigest()
    assert new["metadata"]["derived_from"] == old["checkpoint_id"]
    assert new["metadata"]["checkpoint_id"] == "run-id"
    assert new["metadata"]["opponent_members"] == [
        {"reference": "training-opponents/models/opp"}
    ]
    assert (
        publish.resign_description(_description("tdm-alpha"), "dataset", PATHS) is None
    )
    with pytest.raises(publish.PublishError, match="does not match"):
        publish.resign_description(raw.replace(b'"d"', b'"e"'), "dataset", PATHS)


def test_clean_model_file_keeps_program_weights_and_memory(tmp_path: Path) -> None:
    from marl_battlegrounds import portable_models

    parts = {
        "decide.jaxexport": b"program",
        "weights.npz": b"weights /home/x/ by chance",
        "initial_memory.npz": b"memory",
        "actor_details.json": _description(f"{SEARCH}/opponents/models/opp"),
    }
    source = portable_models._write(
        tmp_path / "old.zip",
        {"checkpoint_id": json.loads(parts["actor_details.json"])["checkpoint_id"]},
        parts,
    )
    assert publish.clean_model_file(source, tmp_path / "new.zip", PATHS)
    with zipfile.ZipFile(tmp_path / "new.zip") as archive:
        new = {name: archive.read(name) for name in archive.namelist()}
    for name in ("decide.jaxexport", "weights.npz", "initial_memory.npz"):
        assert new[name] == parts[name]
    model = json.loads(new["model.json"])
    details = json.loads(new["actor_details.json"])
    assert model["checkpoint_id"] == details["checkpoint_id"]
    assert (
        model["parts"]["actor_details.json"]
        == hashlib.sha256(new["actor_details.json"]).hexdigest()
    )
    assert "/home/" not in new["actor_details.json"].decode()


def test_cards_cite_the_paper_and_take_its_arxiv_number(paper_readme: Path) -> None:
    entry, arxiv = publish.paper_citation(paper_readme.read_text())
    assert (entry, arxiv) == (PAPER, None)
    with pytest.raises(publish.PublishError, match="software"):
        publish.paper_citation("## Citation\n\n```bibtex\n@software{x,\n}\n```\n")
    section = publish.citation_section(entry, None)
    card = f"---\ntags:\n- jax\n---\n\n# X\n\n{section}\n## License\n\nApache.\n"
    numbered = PAPER.replace("arXiv preprint", "arXiv preprint arXiv:2610.01234")
    new = publish.with_citation(card, numbered, "2610.01234")
    assert "tags:\n- arxiv:2610.01234\n- jax\n" in new
    assert "Paper: https://arxiv.org/abs/2610.01234" in new and numbered in new
    assert new.endswith("## License\n\nApache.\n")
    assert publish.with_citation(new, numbered, "2610.01234") == new


def test_readme_citation_matches_citation_cff() -> None:
    root = Path(publish.__file__).resolve().parents[2]
    entry, _ = publish.paper_citation((root / "README.md").read_text())
    preferred = (root / "CITATION.cff").read_text().split("preferred-citation:", 1)[1]
    title = re.search(r"^\s+title: \"(.+)\"$", preferred, re.MULTILINE)
    bib_title = re.search(r"title\s*=\s*\{(.+)\},", entry)
    bib_authors = re.search(r"author\s*=\s*\{(.+)\},", entry)
    assert title and bib_title and bib_authors
    assert bib_title.group(1).replace("{", "").replace("}", "") == title.group(1)
    given = re.findall(r'given-names: "(.+)"', preferred)
    family = re.findall(r'family-names: "(.+)"', preferred)
    expected = [f"{last}, {first}" for first, last in zip(given, family, strict=True)]
    assert bib_authors.group(1).split(" and ") == expected
