"""Check saved-array cleanup without changing learner work or current state.

Tiny CPU Orbax saves expose the real reference cycle. Lightweight runners check
write marking, cleanup after both update branches, and public call boundaries.
These tests check resource lifetimes, not learner mathematics or GPU costs.
"""

from __future__ import annotations

import gc
import time
import weakref
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.training import checkpoints, runner

# pyright: reportPrivateUsage=false


class _Box:
    carry: object
    cycle: _Box


class _BoundaryReachedError(Exception):
    pass


@contextmanager
def _manual_collection() -> Generator[None]:
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def _owner(tmp_path: Path) -> Any:  # noqa: ANN401
    owner = object.__new__(runner._Run)
    owner.root = tmp_path
    owner.host = {"save_seconds": 0.0}
    owner._checkpoint_cleanup_pending = False
    return owner


def test_real_save_cleanup_releases_old_array_and_keeps_current_array(
    tmp_path: Path,
) -> None:
    owner = _owner(tmp_path)
    with _manual_collection():
        owner.state = jnp.arange(8, dtype=jnp.float32)
        for generation in (1, 2):
            old = owner.state
            reference = weakref.ref(old)
            checkpoints._save_arrays(tmp_path / f"arrays-{generation}", {"old": old})
            current = jnp.arange(8, dtype=jnp.float32) + generation * 10
            owner.state = current
            del old
            assert reference() is not None
            owner._checkpoint_cleanup_pending = True
            owner._collect_checkpoint_garbage()
            assert reference() is None
            np.testing.assert_array_equal(owner.state, np.arange(8) + generation * 10)
            assert owner.state is current
            assert owner.host["save_seconds"] >= 0


def test_cleanup_coalesces_pending_writes_and_skips_ordinary_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _owner(tmp_path)
    calls: list[str] = []
    ticks = iter((10.0, 10.25, 20.0, 20.5))
    monkeypatch.setattr(runner.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    owner._collect_checkpoint_garbage()
    assert calls == []
    for _ in range(3):
        owner._checkpoint_cleanup_pending = True
    owner._collect_checkpoint_garbage()
    owner._collect_checkpoint_garbage()
    assert calls == ["collect"]
    assert owner.host["save_seconds"] == 0.25
    owner._checkpoint_cleanup_pending = True
    owner._collect_checkpoint_garbage()
    assert calls == ["collect", "collect"]
    assert owner.host["save_seconds"] == 0.75
    assert not owner._checkpoint_cleanup_pending


def _noop(*args: object, **kwargs: object) -> None:
    pass


def _writer_owner(tmp_path: Path) -> Any:  # noqa: ANN401
    owner = _owner(tmp_path)
    settings = SimpleNamespace(
        input_scale=1.0, spawn_frame="left", parameter_sharing="all"
    )
    owner.config = SimpleNamespace(
        method="mappo", ppo=settings, seed=1, num_envs=4, retain_all_actors=False
    )
    owner.qmix = owner.pqn = None
    owner.state = SimpleNamespace(
        carry=SimpleNamespace(
            opponent_selection=None,
            partner_selection=None,
            history=SimpleNamespace(
                current_variables={"weight": np.ones(1)},
                historical_variables={"weight": np.ones((1, 1))},
            ),
        )
    )
    owner.collection = SimpleNamespace(
        partner_records=(),
        opponent_records=(),
        pinned_opponent=None,
        binding=SimpleNamespace(model_dump=Mock(return_value={})),
    )
    owner.metadata = {"run_id": "memory-test"}
    owner.host.update(env_steps=4, actors={}, past_copies={}, saves=0)
    owner.checkpoint = tmp_path / "checkpoint"
    owner.checkpoint_id = "a" * 64
    owner.prior_elapsed = 0.0
    owner.start = time.monotonic()
    owner.writer = None
    owner.status = {}
    owner._snapshot_exposure = _noop
    owner._used_opponent_members = Mock(return_value=[])
    owner._used_partner_members = Mock(return_value=[])
    owner.set_status = owner.event = owner.prune_recovery = _noop
    return owner


@pytest.mark.parametrize("write", ("save", "actor", "history"))
@pytest.mark.parametrize("fails", (False, True))
def test_writers_mark_cleanup_before_success_or_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write: str, fails: bool
) -> None:
    owner = _writer_owner(tmp_path)
    calls: list[str] = []

    def write_arrays(*args: object, **kwargs: object) -> Path:
        assert owner._checkpoint_cleanup_pending
        calls.append(write)
        if fails:
            raise OSError("Partial checkpoint write")
        return tmp_path / "checkpoint"

    monkeypatch.setattr(checkpoints, "save_checkpoint", write_arrays)
    monkeypatch.setattr(checkpoints, "export_system", write_arrays)
    monkeypatch.setattr(
        checkpoints,
        "read_checkpoint_description",
        Mock(return_value={"checkpoint_id": "a" * 64, "metadata": {}}),
    )
    action = (
        owner.save
        if write == "save"
        else owner.actor
        if write == "actor"
        else lambda: owner._export_past(
            SimpleNamespace(
                created=True, capture_id=0, rounds=1, slot=0, update_index=1
            )
        )
    )
    if fails:
        with pytest.raises(OSError, match="Partial checkpoint write"):
            action()
    else:
        action()
    assert calls == [write]
    assert owner._checkpoint_cleanup_pending


@pytest.mark.parametrize("write", ("actor", "history"))
def test_reused_exports_do_not_mark_new_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write: str
) -> None:
    owner = _writer_owner(tmp_path)
    existing = (
        tmp_path / "actors" / "existing"
        if write == "actor"
        else tmp_path / "past_copies" / "mappo_past_step_4_capture_000000"
    )
    existing.mkdir(parents=True)
    monkeypatch.setattr(checkpoints, "artifact_directory", Mock(return_value=existing))
    monkeypatch.setattr(
        checkpoints,
        "read_checkpoint_description",
        Mock(return_value={"checkpoint_id": "a" * 64, "metadata": {}}),
    )
    monkeypatch.setattr(runner, "_checked_export_provenance", Mock(return_value={}))
    calls: list[bool] = []

    def reuse(*args: object, **kwargs: object) -> None:
        calls.append(owner._checkpoint_cleanup_pending)

    monkeypatch.setattr(checkpoints, "export_system", reuse)
    if write == "actor":
        owner.actor()
    else:
        owner._export_past(
            SimpleNamespace(
                created=True, capture_id=0, rounds=1, slot=0, update_index=1
            )
        )
    assert calls == [False]
    assert not owner._checkpoint_cleanup_pending


@pytest.mark.parametrize(
    "method", ("mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn")
)
@pytest.mark.parametrize("resumed", (False, True))
def test_update_branches_drop_old_owners_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, resumed: bool
) -> None:
    from marl_battlegrounds.training import analysis, collection

    owner = _owner(tmp_path)
    owner.config = SimpleNamespace(
        method=method,
        total_env_steps=2,
        checkpoint_interval_updates=10,
        ppo=SimpleNamespace(value_normalization=True),
    )
    owner.qmix = object() if method == "qmix" else None
    owner.pqn = object() if method == "pqn_vdn" else None
    owner.ppo_reporting = SimpleNamespace(actor_lr=0.001)
    owner.host.update(
        env_steps=0,
        completed_updates=0,
        training_seconds=0.0,
        actor_decisions=0,
        used_policy_samples=0,
        used_value_samples=0,
    )
    owner.metadata = {"execution": {}, "attempt_id": "memory-test"}
    owner.prior_elapsed = 0.0
    owner.start = time.monotonic()
    owner.created_at = time.time()
    owner.runtime = {}
    owner.writer = owner.speed_estimate = None
    owner.checkpoint = tmp_path / "checkpoint" if resumed else None
    owner.validation_steps = owner.random_steps = ()
    owner.collection = object()
    owner.state = _Box()
    owner.state.carry = object()
    references = {"old_state": weakref.ref(owner.state)}
    owner.event = owner.set_status = owner._choose_next_games = _noop
    owner._count_ppo_exposure = owner._count_member_results = _noop
    owner._count_qmix_block = owner._count_pqn_block = _noop
    owner._export_past = owner._write_update = _noop
    owner.block_length = lambda: 1
    owner._qmix_row = owner._pqn_row = Mock(return_value={})

    def saved_boundary() -> None:
        owner._checkpoint_cleanup_pending = True

    owner.save = saved_boundary
    owner.finish_pending = saved_boundary if resumed else _noop
    summary = SimpleNamespace(
        real_transitions=1,
        live_actor_decisions=1,
        stage_completed=[0],
        stage_score_sums=[0],
        stage_length_sum=[0],
        stage_k20_count=[0],
        task_reward_sum=0.0,
        active_samples=1,
        shaping_reward_sum=0.0,
    )
    metrics = SimpleNamespace(actor_samples=np.zeros(1), critic_samples=np.zeros(1))
    host_result = SimpleNamespace(
        performed=True,
        accepted=True,
        failed=False,
        summary=summary,
        metrics=metrics,
        snapshot=None,
    )

    def collect(*args: object, **kwargs: object) -> tuple[_Box, _Box]:
        collected, rollout = _Box(), _Box()
        references.update(
            collected=weakref.ref(collected), rollout=weakref.ref(rollout)
        )
        return collected, rollout

    def update(*args: object) -> tuple[_Box, _Box]:
        state, result = _Box(), _Box()
        state.carry = object()
        references.update(current=weakref.ref(state), result=weakref.ref(result))
        return state, result

    def host_transfer(value: object) -> SimpleNamespace:
        return host_result

    owner.updater = update
    monkeypatch.setattr(collection, "collect_training_rollout", collect)
    monkeypatch.setattr(jax, "device_get", host_transfer)
    monkeypatch.setattr(runner, "_memory_snapshot", Mock(return_value={}))
    monkeypatch.setattr(analysis, "refresh_summary", _noop)
    checked: list[bool] = []

    def check_boundary() -> None:
        assert owner._checkpoint_cleanup_pending
        assert owner.host["env_steps"] == 1
        assert all(
            references[name]() is None
            for name in ("old_state", "collected", "rollout", "result")
        )
        assert references["current"]() is owner.state
        runner._Run._collect_checkpoint_garbage(owner)
        assert references["current"]() is owner.state
        checked.append(True)
        raise _BoundaryReachedError

    owner._collect_checkpoint_garbage = check_boundary
    with pytest.raises(_BoundaryReachedError):
        owner.execute()
    assert checked == [True]
    assert not owner._checkpoint_cleanup_pending


def _public_call(kind: str, tmp_path: Path) -> object:
    if kind == "train":
        return runner.train(output_dir=tmp_path)
    return runner.extend_training(
        tmp_path / "checkpoint", additional_env_steps=1, output_dir=tmp_path / "child"
    )


@pytest.mark.parametrize("kind", ("train", "extend"))
def test_public_call_cleans_before_setup_and_after_inner_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    references: list[weakref.ReferenceType[object]] = []
    result = runner.TrainResult(
        tmp_path,
        tmp_path / "actor",
        None,
        1,
        1,
        "complete",
        (),
        tmp_path / "checkpoint",
    )
    with _manual_collection():
        prior = _Box()
        prior.cycle = prior
        prior_reference = weakref.ref(prior)
        del prior
        assert prior_reference() is not None

        def train(*args: object, **kwargs: object) -> object:
            assert prior_reference() is None
            owner = _owner(tmp_path)
            owner.state = jnp.arange(8, dtype=jnp.float32)
            checkpoints._save_arrays(tmp_path / "arrays", {"state": owner.state})
            references.extend((weakref.ref(owner), weakref.ref(owner.state)))
            return result

        monkeypatch.setattr(runner, "_train", train)
        assert _public_call(kind, tmp_path) is result
        assert all(reference() is None for reference in references)


@pytest.mark.parametrize("kind", ("train", "extend"))
def test_public_failure_preserves_traceback_and_next_call_cleans_released_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    references: list[weakref.ReferenceType[_Box]] = []
    collected: list[bool] = []
    original_collect = gc.collect

    def collect() -> int:
        collected.append(True)
        return original_collect()

    def fail(*args: object, **kwargs: object) -> None:
        value = _Box()
        value.cycle = value
        references.append(weakref.ref(value))
        raise OSError("Keep this traceback")

    def failing_call() -> None:
        with pytest.raises(OSError, match="Keep this traceback") as caught:
            _public_call(kind, tmp_path)
        assert len(collected) == 2
        assert caught.value.__traceback__ is not None
        assert references[0]() is not None

    with _manual_collection():
        monkeypatch.setattr(runner.gc, "collect", collect)
        monkeypatch.setattr(runner, "_train", fail)
        failing_call()

        def success(*args: object, **kwargs: object) -> None:
            assert references[0]() is None

        monkeypatch.setattr(runner, "_train", success)
        _public_call(kind, tmp_path)
        assert len(collected) == 4
