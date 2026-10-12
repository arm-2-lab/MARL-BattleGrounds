"""Check which BattleClient call folders ``remove_unplayed_evidence`` deletes.

A match folder (the ``<replay>.model-calls`` folder or a ``generation-N`` folder
below it) goes only when ``session.json`` is its one entry and records
``accepted_turns`` 0 and no ``saved_prefixes``; the top folder goes too when
nothing else is left. A folder with a played turn, a saved prefix, a policy
trace or any other file stays, and so does one whose ``session.json`` cannot be
read. A missing folder does nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from marl_battlegrounds._battle_client.system_worker import remove_unplayed_evidence


def _match(
    folder: Path,
    *,
    turns: int = 0,
    prefixes: list[dict[str, object]] | None = None,
    extra: str | None = None,
) -> Path:
    folder.mkdir(parents=True)
    details = {"accepted_turns": turns, "saved_prefixes": prefixes or []}
    (folder / "session.json").write_text(json.dumps(details), encoding="utf-8")
    if extra is not None:
        (folder / extra).write_text("{}\n", encoding="utf-8")
    return folder


def test_an_unplayed_match_leaves_nothing(tmp_path: Path) -> None:
    folder = _match(tmp_path / "battle_1.marlbg-replay.json.model-calls")
    remove_unplayed_evidence(folder)
    assert not folder.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_saved_game_keeps_its_folder_but_not_an_unplayed_next_match(
    tmp_path: Path,
) -> None:
    saved: dict[str, object] = {"replay_id": "r", "path": "p", "transitions": 40}
    folder = _match(
        tmp_path / "battle_2.marlbg-replay.json.model-calls",
        turns=40,
        prefixes=[saved],
        extra="policy_trace.jsonl",
    )
    _match(folder / "generation-1")
    remove_unplayed_evidence(folder)
    assert sorted(entry.name for entry in folder.iterdir()) == [
        "policy_trace.jsonl",
        "session.json",
    ]


def test_played_or_called_matches_stay(tmp_path: Path) -> None:
    played = _match(tmp_path / "played", turns=3)
    saved = _match(tmp_path / "saved", prefixes=[{"replay_id": "r"}])
    called = _match(tmp_path / "called", extra="model_calls.jsonl")
    for folder in (played, saved, called):
        remove_unplayed_evidence(folder)
        assert (folder / "session.json").is_file()


def test_an_unreadable_record_stays(tmp_path: Path) -> None:
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "session.json").write_text("{not json", encoding="utf-8")
    remove_unplayed_evidence(folder)
    assert (folder / "session.json").is_file()


def test_a_missing_folder_does_nothing(tmp_path: Path) -> None:
    remove_unplayed_evidence(tmp_path / "never-made")
    assert list(tmp_path.iterdir()) == []
