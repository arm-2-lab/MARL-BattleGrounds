"""Check that a new or child training run owns its folder before it writes there.

train and extend_training check early that output_dir is new or empty. A call
can then wait in setup while another call claims the same folder. These CPU
tests hold a second public call just after that early check, let a first call
publish, then release the second call and compare every path and file hash of
the first run before and after:

- A delayed fresh call stops with ValueError once the first call has saved its
  run and exited: with no panels, with a new validation panel against a run
  without one (no panel is written into that run), and with the same panel.
- A delayed extend_training child stops the same way; the parent is unchanged.
- A fresh call with a new validation panel stops with RuntimeError while the
  first call still holds the run lock, and writes no panel into the live run.
- A delayed resume of a run folder stops with ValueError once another resume
  has saved a newer checkpoint, so the run is never rolled back.
- A folder holding only ``.training.lock``, left by a call that stopped before
  saving anything, counts as empty for train and for an extend_training child,
  and the lock file stays empty after the run. A folder with any other entry,
  or with a lock entry that is not a plain file, is refused before setup and
  left unchanged. _run_io.unused_run_folder, the one shared rule, gives the
  same answers.

Learner execution is stubbed after one step-0 checkpoint save, so a first run
looks like a real stopped run. These tests prove folder ownership only, not
training results or GPU behavior.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import hashlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests.helpers.training_continuation_helpers import fixed_source, latest

from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.training import TrainConfig, extend_training, runner, train
from marl_battlegrounds.training._run_io import unused_run_folder

CONFIG = TrainConfig(
    method="ff_ippo",
    keep_past=0,
    num_envs=2,
    total_env_steps=4,
    curriculum=[{"share": 1.0, "maps": [0], "team_size": 2}],
    ppo=PPOConfig(rollout_length=2, epochs=1),
    metrics="none",
    verbose=False,
)

Hold = tuple[threading.Event, threading.Event]


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else "folder"
        )
        for path in sorted(root.rglob("*"))
    }


def _stub_learner(
    monkeypatch: pytest.MonkeyPatch,
    hold: Hold | None = None,
    *,
    always_save: bool = False,
) -> None:
    def execute(owner: runner._Run) -> Path:
        if owner.checkpoint is None or always_save:
            owner.save()
        if hold is not None and threading.current_thread().name == "caller-a":
            # The first call has published its run and still holds the lock.
            hold[0].set()
            hold[1].wait()
        return owner.root

    monkeypatch.setattr(runner._Run, "execute", execute)


def _race(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    first: Callable[[], object],
    second: Callable[[], object],
    hold: Hold | None = None,
) -> tuple[dict[str, str], dict[str, str], object]:
    bind = runner._bind_training_population
    admitted, release = threading.Event(), threading.Event()

    def held(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        # The first call after the early output_dir check holds caller B there.
        if threading.current_thread().name == "caller-b" and args[0] == "opponent":
            admitted.set()
            release.wait()
        return bind(*args, **kwargs)

    monkeypatch.setattr(runner, "_bind_training_population", held)
    outcome: dict[str, object] = {}

    def run(name: str, call: Callable[[], object]) -> None:
        try:
            outcome[name] = call()
        except Exception as error:
            outcome[name] = error
        finally:
            # A call that fails early must not leave the main thread waiting.
            if name == "b":
                admitted.set()
            elif hold is not None:
                hold[0].set()

    caller_b = threading.Thread(target=run, args=("b", second), name="caller-b")
    caller_b.start()
    admitted.wait()
    caller_a = threading.Thread(target=run, args=("a", first), name="caller-a")
    caller_a.start()
    if hold is None:
        caller_a.join()
    else:
        hold[0].wait()
    before = _snapshot(root)
    release.set()
    caller_b.join()
    after = _snapshot(root)
    if hold is not None:
        hold[1].set()
        caller_a.join()
    assert outcome["a"] == root
    assert {"run_details.json", "latest_checkpoint.json", "status.json"} <= set(before)
    return before, after, outcome["b"]


@pytest.mark.parametrize(
    ("first_panel", "second_panel"),
    [(False, False), (False, True), (True, True)],
    ids=["no-panel", "new-panel", "same-panel"],
)
def test_delayed_fresh_call_cannot_replace_a_finished_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_panel: bool,
    second_panel: bool,
) -> None:
    fixed_source(monkeypatch, "ff_ippo")
    _stub_learner(monkeypatch)
    root = tmp_path / "run"

    def fresh(panel: bool) -> Callable[[], object]:
        opponents = ["random"] if panel else None
        return lambda: train(CONFIG, output_dir=root, validation_opponents=opponents)

    before, after, second = _race(
        monkeypatch, root, fresh(first_panel), fresh(second_panel)
    )
    assert isinstance(second, ValueError)
    assert "must be new or empty" in str(second)
    assert "while this call was setting up" in str(second)
    assert after == before
    assert ("validation_panel/panel.json" in before) is first_panel


def test_delayed_child_call_cannot_replace_a_finished_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_source(monkeypatch, "ff_ippo")
    _stub_learner(monkeypatch)
    parent = tmp_path / "parent"
    assert train(CONFIG, output_dir=parent) == parent
    checkpoint = latest(parent)
    parent_before = _snapshot(parent)
    root = tmp_path / "child"

    def child() -> object:
        return extend_training(checkpoint, additional_env_steps=2, output_dir=root)

    before, after, second = _race(monkeypatch, root, child, child)
    assert isinstance(second, ValueError)
    assert "must be new or empty" in str(second)
    assert "while this call was setting up" in str(second)
    assert after == before
    assert _snapshot(parent) == parent_before


def test_fresh_call_writes_nothing_while_another_call_holds_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_source(monkeypatch, "ff_ippo")
    hold = (threading.Event(), threading.Event())
    _stub_learner(monkeypatch, hold)
    root = tmp_path / "run"
    before, after, second = _race(
        monkeypatch,
        root,
        lambda: train(CONFIG, output_dir=root),
        lambda: train(CONFIG, output_dir=root, validation_opponents=["random"]),
        hold,
    )
    assert isinstance(second, RuntimeError)
    assert "already has an active writer" in str(second)
    assert after == before
    assert not (root / "validation_panel").exists()


def test_delayed_resume_cannot_roll_a_run_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_source(monkeypatch, "ff_ippo")
    _stub_learner(monkeypatch, always_save=True)
    root = tmp_path / "run"
    assert train(CONFIG, output_dir=root) == root
    first_checkpoint = latest(root)

    def resume() -> object:
        return train(resume_from=root)

    before, after, second = _race(monkeypatch, root, resume, resume)
    assert latest(root) != first_checkpoint
    assert isinstance(second, ValueError)
    assert "newer checkpoint" in str(second)
    assert after == before


def test_lock_only_folder_is_used_and_other_entries_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_source(monkeypatch, "ff_ippo")
    _stub_learner(monkeypatch)
    outside = tmp_path / "outside.lock"
    outside.touch()
    for occupant in ("file", "lock-and-run", "lock-folder", "lock-link"):
        folder = tmp_path / occupant
        folder.mkdir()
        lock = folder / ".training.lock"
        if occupant == "file":
            (folder / "notes.txt").write_text("Keep me")
        elif occupant == "lock-and-run":
            lock.touch()
            (folder / "run_details.json").write_text("{}\n")
        elif occupant == "lock-folder":
            lock.mkdir()
        else:
            lock.symlink_to(outside)
        expected = _snapshot(folder)
        assert not unused_run_folder(folder)
        with pytest.raises(ValueError, match=r"^output_dir must be new or empty$"):
            train(CONFIG, output_dir=folder)
        assert _snapshot(folder) == expected
    assert unused_run_folder(tmp_path / "missing")
    assert not unused_run_folder(outside)

    root = tmp_path / "run"
    root.mkdir()
    assert unused_run_folder(root)
    (root / ".training.lock").touch()
    assert unused_run_folder(root)
    assert train(CONFIG, output_dir=root) == root
    snapshot = _snapshot(root)
    assert snapshot[".training.lock"] == hashlib.sha256(b"").hexdigest()
    assert {"run_details.json", "latest_checkpoint.json", "status.json"} <= set(
        snapshot
    )

    child = tmp_path / "child"
    child.mkdir()
    (child / ".training.lock").touch()
    assert (
        extend_training(latest(root), additional_env_steps=2, output_dir=child) == child
    )
    assert (child / "run_details.json").is_file()
