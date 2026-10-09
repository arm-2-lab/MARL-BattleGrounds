"""Check the record of whether XLA's deterministic GPU mode was declared.

Contracts: declared_gpu_deterministic_ops reads the last
--xla_gpu_deterministic_ops setting (bare or =true on, =false off, any other
spelling unknown, absent off); capture_execution_settings records this
attempt's whole XLA_FLAGS string; merge_execution_settings keeps one item per
attempt and gives False if any is off, otherwise None if any is unknown or
unrecorded, otherwise True. A RunWriter resumed under different flags succeeds; a
resume that saves no game leaves run_details.json byte for byte unchanged; an
attempt's settings join the record once, when it saves its first game of the
pass, so a first attempt that saved no game leaves nothing; and an earlier
attempt that saved games without a record counts as unknown. Saved results
merge every pass with games, and passes without a record make the answer
unknown; canonical results add the source passes of
reused games, compute nothing until the property is read and hand out a copy.
Training's declared_xla_flags value is the same string. No game is played.
"""

# pyright: reportPrivateUsage=false
import json
from pathlib import Path
from typing import cast

import pytest
from tests.evaluation.test_result_access import _fixture, _save

from marl_battlegrounds.evaluation.results import load_results
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.evaluation.runtime_provenance import (
    capture_execution_settings,
    declared_gpu_deterministic_ops,
    merge_execution_settings,
)

ON = "--xla_gpu_deterministic_ops=true"


@pytest.mark.parametrize(
    ("flags", "state"),
    [
        ("", False),
        ("--xla_gpu_deterministic_ops", True),
        (ON, True),
        ("--xla_gpu_deterministic_ops=false", False),
        (f"{ON} --xla_gpu_deterministic_ops=false", False),
        (f"--xla_gpu_deterministic_ops=false  {ON}", True),
        ("--xla_gpu_deterministic_ops=1", None),
        ("--xla_gpu_deterministic_ops=True", None),
        ("--xla_gpu_autotune_level=0", False),
    ],
)
def test_flag_parsing(flags: str, state: bool | None) -> None:
    assert declared_gpu_deterministic_ops(flags) is state


def test_capture_records_the_whole_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLA_FLAGS", f"{ON}  --xla_dump_to=/tmp/x")
    assert capture_execution_settings() == {
        "declared_xla_flags_by_attempt": [f"{ON}  --xla_dump_to=/tmp/x"],
        "declared_gpu_deterministic_ops": True,
    }
    monkeypatch.delenv("XLA_FLAGS")
    assert capture_execution_settings() == {
        "declared_xla_flags_by_attempt": [""],
        "declared_gpu_deterministic_ops": False,
    }


def _settings(flags: str) -> dict[str, object]:
    return {
        "declared_xla_flags_by_attempt": [flags],
        "declared_gpu_deterministic_ops": declared_gpu_deterministic_ops(flags),
    }


def test_merge_rule() -> None:
    on, off = _settings(ON), _settings("")
    assert merge_execution_settings([on, on]) == {
        "declared_xla_flags_by_attempt": [ON, ON],
        "declared_gpu_deterministic_ops": True,
    }
    assert (
        merge_execution_settings([on, None])["declared_gpu_deterministic_ops"] is None
    )
    assert merge_execution_settings([None, on])["declared_xla_flags_by_attempt"] == [
        None,
        ON,
    ]
    assert (
        merge_execution_settings([None, off, on])["declared_gpu_deterministic_ops"]
        is False
    )
    assert merge_execution_settings([]) == {
        "declared_xla_flags_by_attempt": [],
        "declared_gpu_deterministic_ops": None,
    }


def _saved_settings(path: Path) -> dict[str, object]:
    manifest = json.loads((path / "run_details.json").read_bytes())
    entry = next(iter(manifest["passes"].values()))
    return cast(dict[str, object], entry["details"]["execution_settings"])


def _save_game(writer: RunWriter, episode_id: int) -> None:
    # Stands in for a completed game: only the commit step is under test.
    writer._pending.append((writer._pass_key, episode_id))
    writer.flush()


def test_resume_without_games_rewrites_nothing(tmp_path: Path) -> None:
    with RunWriter(tmp_path, details={"execution_settings": _settings(ON)}) as first:
        path = first.run_dir
        _save_game(first, 1)
    before = (path / "run_details.json").read_bytes()
    with RunWriter(resume_from=path, details={"execution_settings": _settings("")}):
        pass
    assert (path / "run_details.json").read_bytes() == before


def test_resumed_attempt_with_games_joins_the_record(tmp_path: Path) -> None:
    with RunWriter(tmp_path, details={"execution_settings": _settings(ON)}) as first:
        path = first.run_dir
        _save_game(first, 1)
    with RunWriter(
        resume_from=path, details={"execution_settings": _settings("")}
    ) as resumed:
        resumed.start_pass(
            phase="evaluation",
            pass_id="1",
            details={"execution_settings": _settings("")},
        )
        _save_game(resumed, 2)
        _save_game(resumed, 3)
    assert _saved_settings(path) == {
        "declared_xla_flags_by_attempt": [ON, ""],
        "declared_gpu_deterministic_ops": False,
    }


def test_attempt_without_games_leaves_no_record(tmp_path: Path) -> None:
    with RunWriter(tmp_path, details={"execution_settings": _settings("")}) as first:
        path = first.run_dir
    with RunWriter(
        resume_from=path, details={"execution_settings": _settings(ON)}
    ) as resumed:
        _save_game(resumed, 1)
    assert _saved_settings(path) == {
        "declared_xla_flags_by_attempt": [ON],
        "declared_gpu_deterministic_ops": True,
    }


def test_earlier_attempt_with_games_and_no_record_is_unknown(tmp_path: Path) -> None:
    with RunWriter(tmp_path) as first:
        path = first.run_dir
        _save_game(first, 1)
    with RunWriter(
        resume_from=path, details={"execution_settings": _settings(ON)}
    ) as resumed:
        _save_game(resumed, 2)
    assert _saved_settings(path) == {
        "declared_xla_flags_by_attempt": [None, ON],
        "declared_gpu_deterministic_ops": None,
    }


@pytest.mark.parametrize(
    ("first", "second", "state"),
    [
        (_settings(ON), _settings(ON), True),
        (_settings(ON), None, None),
        (_settings(ON), _settings(""), False),
    ],
)
def test_results_merge_every_pass_with_games(
    tmp_path: Path,
    first: dict[str, object] | None,
    second: dict[str, object] | None,
    state: bool | None,
) -> None:
    directory, manifest = _fixture(tmp_path, modes=("none", "none"))
    for entry, record in zip(manifest["passes"].values(), (first, second), strict=True):
        if record is not None:
            entry["details"]["execution_settings"] = record
    _save(directory, manifest)
    settings = load_results(directory).execution_settings
    assert settings["declared_gpu_deterministic_ops"] is state
    assert len(settings["declared_xla_flags_by_attempt"]) == 2  # pyright: ignore[reportArgumentType]


def test_training_identity_reads_the_same_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.training._compilation import execution_identity

    monkeypatch.setenv("XLA_FLAGS", f"{ON} --xla_force_host_platform_device_count=1")
    runtime = {
        "backend": "cpu",
        "device": "cpu",
        "runtime_version": None,
        "precision": "float32",
    }
    identity = execution_identity(runtime=runtime)
    assert identity["declared_xla_flags"] == (
        f"{ON} --xla_force_host_platform_device_count=1"
    )


def test_canonical_results_merge_source_passes_only_when_asked(
    tmp_path: Path,
) -> None:
    from tests.evaluation.test_system_metrics import _canonical

    path, _, _ = _canonical(tmp_path)
    results = load_results(path)
    view = results._view
    assert not hasattr(view, "_settings_cache")
    settings = results.execution_settings
    assert hasattr(view, "_settings_cache")
    # The fixture's source passes saved no record: every one is unknown.
    attempts = settings["declared_xla_flags_by_attempt"]
    assert isinstance(attempts, list) and attempts and set(attempts) == {None}  # pyright: ignore[reportUnknownArgumentType]
    assert settings["declared_gpu_deterministic_ops"] is None
    settings["declared_xla_flags_by_attempt"] = []
    assert results.execution_settings["declared_xla_flags_by_attempt"] == attempts
