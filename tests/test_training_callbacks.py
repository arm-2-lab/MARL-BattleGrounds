"""Check optional callbacks, cleanup and durable factory receipts without an SDK.

The fake W&B run proves axis setup, attempt separation and cleanup. Subprocess
checks forbid optional/numerical imports when logging is disabled. Scoped
factories reuse ordinary trainer keyword routes; they do not own a scheduler.
"""

from __future__ import annotations

# The factories and metadata reader are shared private owner seams.
# pyright: reportPrivateUsage=false
import copy
import json
import os
import subprocess
import sys
from collections.abc import Callable, Generator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from marl_battlegrounds.training import make_wandb_callback, study


class FakeRun:
    def __init__(self, *, fail: bool = False) -> None:
        self.axes: list[tuple[str, str | None]] = []
        self.rows: list[dict[str, Any]] = []
        self.finished = 0
        self.fail = fail

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        self.axes.append((name, step_metric))

    def log(self, data: Mapping[str, Any]) -> None:
        self.rows.append(dict(data))
        if self.fail:
            raise LookupError("SDK write failed")

    def finish(self) -> None:
        self.finished += 1


def test_adapter_uses_explicit_axis_and_leaves_lifecycle_with_caller() -> None:
    run = FakeRun()
    callback = make_wandb_callback(run)
    row = {"env_steps": 32, "run_id": "run", "attempt_id": "attempt", "loss": None}
    callback(row)
    assert run.axes == [("env_steps", None), ("*", "env_steps")]
    assert run.rows == [row] and run.rows[0] is not row
    assert run.finished == 0
    with pytest.raises(TypeError, match="define_metric and log"):
        make_wandb_callback(object())  # pyright: ignore[reportArgumentType]


def test_disabled_callback_module_imports_no_sdk_or_backend(tmp_path: Path) -> None:
    program = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'wandb', 'jax', 'numpy', 'flax'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from marl_battlegrounds.training import make_wandb_callback
from marl_battlegrounds.experiments.callbacks import make_pfsp
from marl_battlegrounds.training.study import _runtime_paths, _runtime_callbacks
from contextlib import ExitStack
from pathlib import Path
with ExitStack() as stack:
    assert _runtime_callbacks(_runtime_paths({}, Path.cwd()), stack) == {}
assert make_pfsp()({'members': []}, 0) == {'opponents': {'self': 1.0}}
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure", [False, True])
def test_wandb_scope_groups_attempts_and_closes_after_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    from marl_battlegrounds.experiments.callbacks import wandb_logging

    calls: list[dict[str, Any]] = []
    runs: list[FakeRun] = []

    def init(**kwargs: object) -> FakeRun:
        calls.append(kwargs)
        run = FakeRun(fail=failure)
        runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    monkeypatch.setenv("WANDB_API_KEY", "SECRET_NOT_FOR_RECORDS")
    for attempt in ("first", "resumed"):
        with (
            pytest.raises(LookupError, match="SDK write") if failure else nullcontext(),
            wandb_logging(
                project="software-check", directory=str(tmp_path), mode="offline"
            ) as callback,
        ):
            assert len(calls) == (0 if attempt == "first" else 1)
            row = {"run_id": "durable", "attempt_id": attempt, "env_steps": 32}
            callback(row)
            callback({**row, "env_steps": 64})
    assert [call["group"] for call in calls] == ["durable", "durable"]
    assert [call["id"] for call in calls] == ["durable-first", "durable-resumed"]
    assert all(
        call["resume"] == "never" and call["reinit"] == "create_new" for call in calls
    )
    assert all(run.finished == 1 for run in runs)
    assert "SECRET" not in json.dumps(calls)
    assert all(run.axes == [("env_steps", None), ("*", "env_steps")] for run in runs)


def test_unused_wandb_scope_does_not_import_or_create_files(tmp_path: Path) -> None:
    from marl_battlegrounds.experiments.callbacks import wandb_logging

    before = sys.modules.get("wandb")
    destination = tmp_path / "logs"
    with (
        pytest.raises(RuntimeError, match="before update"),
        wandb_logging(project="test", directory=str(destination)),
    ):
        raise RuntimeError("before update")
    assert sys.modules.get("wandb") is before
    assert not destination.exists()


@pytest.mark.parametrize(
    "bad",
    [
        {"factory": "wrong"},
        {"factory": "a:b.c"},
        {"factory": "a:b", "kwargs": []},
        {"factory": "a:b", "kwargs": {"x": float("nan")}},
        {"factory": "a:b", "surprise": 1},
    ],
)
def test_bad_callback_declarations_fail_without_importing(
    bad: object, tmp_path: Path
) -> None:
    with pytest.raises((ValueError, TypeError)):
        study._runtime_paths({"on_update": bad}, tmp_path)


@pytest.mark.parametrize("failure", ["none", "body", "later_factory", "not_callable"])
def test_scoped_factories_run_once_and_unwind_in_reverse_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from marl_battlegrounds import _method_loading
    from marl_battlegrounds.training.callbacks import _callback_identities

    events: list[str] = []

    @contextmanager
    def logger(*, label: str) -> Generator[Callable[..., object]]:
        events.append(f"open {label}")
        try:
            yield lambda *args, **kwargs: events.append(f"call {label}")
        finally:
            events.append(f"close {label}")

    def second() -> object:
        events.append("factory second")
        if failure == "later_factory":
            raise LookupError("factory error")
        return 5 if failure == "not_callable" else logger(label="second")

    def installed(reference: str) -> Callable[..., object]:
        return logger if reference == "fixture:logger" else second

    def identity(reference: str) -> dict[str, Any]:
        return {
            "kind": "factory",
            "reference": reference,
            "kwargs": {},
            "external_state": "unknown",
        }

    monkeypatch.setattr(_method_loading, "installed_callable", installed)
    monkeypatch.setattr(_method_loading, "_reference_identity", identity)
    runtime = study._runtime_paths(
        {
            "on_update": {"factory": "fixture:logger", "kwargs": {"label": "first"}},
            "matchmaking": {"factory": "fixture:second"},
            "opponents": {"bot": "tdm-alpha"},
        },
        tmp_path,
    )
    before = copy.deepcopy(runtime)

    def use() -> None:
        with ExitStack() as stack:
            bound = study._runtime_callbacks(runtime, stack)
            receipts = _callback_identities(
                on_update=bound["on_update"], matchmaking=bound["matchmaking"]
            )
            assert receipts["on_update"]["kwargs"] == {"label": "first"}
            assert bound["opponents"] == {"bot": "tdm-alpha"}
            bound["on_update"]({})
            if failure == "body":
                raise RuntimeError("training failed")

    if failure == "none":
        use()
    else:
        with pytest.raises((LookupError, TypeError, RuntimeError)):
            use()
    assert runtime == before
    assert events.count("factory second") == 1
    assert events.count("open first") == events.count("close first") == 1
    if failure in ("none", "body"):
        assert events[-2:] == ["close second", "close first"]
    else:
        assert events[-1] == "close first"


@pytest.mark.parametrize("decorated", [False, True])
def test_callback_code_and_kwargs_have_distinct_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decorated: bool
) -> None:
    from marl_battlegrounds import _method_loading

    module = tmp_path / "callback_provider.py"
    prefix = (
        "from contextlib import contextmanager\n@contextmanager\n" if decorated else ""
    )
    verb = "yield" if decorated else "return"
    module.write_text(prefix + f"def callback():\n    {verb} print\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        first = study._callback_inputs(
            {"on_update": {"factory": "callback_provider:callback", "kwargs": {"a": 1}}}
        )
        second = study._callback_inputs(
            {"on_update": {"factory": "callback_provider:callback", "kwargs": {"a": 2}}}
        )
        assert first != second
        module.write_text(prefix + f"def callback():\n    {verb} len\n")
        sys.modules.pop("callback_provider", None)
        third = study._callback_inputs(
            {"on_update": {"factory": "callback_provider:callback", "kwargs": {"a": 1}}}
        )
        assert first != third
        if decorated:
            assert first["on_update"]["callable"] == third["on_update"]["callable"]
            assert (
                first["on_update"]["wrapped_callables"]
                != third["on_update"]["wrapped_callables"]
            )
        else:
            assert "wrapped_callables" not in first["on_update"]
        assert _method_loading._setup_inputs.get() is None
    finally:
        sys.modules.pop("callback_provider", None)


def test_factory_cannot_mutate_saved_kwargs_or_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds import _method_loading
    from marl_battlegrounds.training.callbacks import _callback_identities

    def factory(*, settings: dict[str, Any]) -> Callable[..., object]:
        settings["secretly_changed"] = True
        return print

    def installed(reference: str) -> Callable[..., object]:
        return factory

    def identity(reference: str) -> dict[str, Any]:
        return {"kind": "factory", "reference": reference, "kwargs": {}}

    monkeypatch.setattr(_method_loading, "installed_callable", installed)
    monkeypatch.setattr(_method_loading, "_reference_identity", identity)
    runtime = study._runtime_paths(
        {
            "on_update": {
                "factory": "module:factory",
                "kwargs": {"settings": {"chosen": 1}},
            }
        },
        tmp_path,
    )
    before = copy.deepcopy(runtime)
    with ExitStack() as stack:
        bound = study._runtime_callbacks(runtime, stack)
        receipt = _callback_identities(on_update=bound["on_update"])
    assert runtime == before
    assert receipt["on_update"]["kwargs"] == before["on_update"]["kwargs"]


@pytest.mark.parametrize("fail", [False, True])
def test_public_hydra_adapter_enters_and_closes_callback_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fail: bool,
) -> None:
    from marl_battlegrounds import _method_loading, training
    from marl_battlegrounds.experiments.hydra import run_training

    events: list[str] = []

    @contextmanager
    def factory() -> Generator[Callable[..., object]]:
        events.append("open")
        try:

            def log(row: object) -> None:
                events.append("saved row")

            yield log
        finally:
            events.append("close")

    def installed(reference: str) -> Callable[..., object]:
        return factory

    def identity(reference: str) -> dict[str, str]:
        return {"kind": "factory", "reference": reference}

    def train(config: object, **kwargs: object) -> SimpleNamespace:
        events.append("train")
        callback = cast(Callable[[dict[str, int]], object], kwargs["on_update"])
        callback({"env_steps": 32})
        if fail:
            raise LookupError("training failed")
        return SimpleNamespace(run_dir=kwargs["output_dir"])

    monkeypatch.setattr(_method_loading, "installed_callable", installed)
    monkeypatch.setattr(_method_loading, "_reference_identity", identity)
    monkeypatch.setattr(training, "train", train)
    settings: dict[str, Any] = {
        "method": "ff_ippo",
        "output_dir": str(tmp_path / "run"),
        "runtime": {"on_update": {"factory": "example:logger"}},
    }
    with pytest.raises(LookupError, match="training failed") if fail else nullcontext():
        run_training(settings, base_dir=tmp_path, launch_dir=tmp_path / "launch")
    assert events == ["open", "train", "saved row", "close"]
    output = capsys.readouterr()
    assert str(tmp_path / "run") in (output.err if fail else output.out)
    assert (tmp_path / "launch/.hydra/training_config.json").is_file()


@pytest.mark.parametrize("kind", ["cycle", "non_callable", "too_long"])
def test_invalid_wrapped_factory_chain_fails_before_call(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from marl_battlegrounds import _method_loading

    def factory() -> None:
        pytest.fail("Identity inspection must not call the factory")

    if kind == "cycle":
        factory.__dict__["__wrapped__"] = factory
    elif kind == "non_callable":
        factory.__dict__["__wrapped__"] = None
    else:
        from functools import partial

        outer = factory
        for _ in range(64):
            wrapper = partial(print)
            wrapper.__dict__["__wrapped__"] = outer
            outer = wrapper
        factory = outer

    def installed(reference: str) -> Callable[..., object]:
        return factory

    monkeypatch.setattr(_method_loading, "installed_callable", installed)
    with pytest.raises((TypeError, ValueError), match="__wrapped__"):
        _method_loading._reference_identity("unused:factory")
