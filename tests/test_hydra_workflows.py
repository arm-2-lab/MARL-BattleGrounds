"""Check optional Hydra bootstrap, CLI compatibility and parser restoration.

These software probes never train, load models or start numerical libraries.
Fresh subprocesses cover entry, help, config, finite sweeps and shell completion.
Later composition and study tests extend this same public launcher boundary.
"""

import argparse
import importlib
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from marl_battlegrounds.experiments import hydra as adapter


@pytest.fixture
def entry(tmp_path: Path) -> Path:
    path = tmp_path / "launch.py"
    path.write_text("""import argparse, json, sys
before = argparse.ArgumentParser._check_help
from marl_battlegrounds.experiments.hydra import prepare_hydra
import hydra
from omegaconf import DictConfig, OmegaConf
prepare_hydra()
@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig):
    print("TASK=" + json.dumps(OmegaConf.to_container(cfg, resolve=True)))
try:
    main()
finally:
    assert getattr(argparse.ArgumentParser, "_check_help") is before  # noqa: B009
    assert not {'jax', 'jaxlib', 'flax', 'optax', 'flashbax'} & sys.modules.keys()
""")
    return path


@pytest.mark.parametrize(
    ("arguments", "expected", "tasks"),
    [
        (["+seed=1"], 'TASK={"seed": 1}', 1),
        (["--help"], "Hydra", 0),
        (["--hydra-help"], "--shell-completion", 0),
        (["--cfg", "job", "+seed=1"], "seed: 1", 0),
        (["--cfg", "job", "--resolve", "+seed=1", "+copy=${seed}"], "copy: 1", 0),
        (["--multirun", "+seed=1,2"], 'TASK={"seed": 2}', 2),
        (["--hydra-help"], "install=bash", 0),
        (["--shell-completion", "install=bash"], "hydra_bash_completion", 0),
        (["--shell-completion", "query=bash"], "hydra.", 0),
    ],
)
def test_hydra_cli_matrix(
    entry: Path, arguments: list[str], expected: str, tasks: int
) -> None:
    output = entry.parent / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(entry),
            *arguments,
            f"hydra.run.dir={output}/run",
            f"hydra.sweep.dir={output}/sweep",
            "hydra.job.chdir=false",
        ],
        cwd=entry.parent,
        env={**os.environ, "COMP_LINE": "python launch.py ", "COMP_POINT": "0"},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout
    assert result.stdout.count("TASK=") == tasks
    if tasks == 0:
        assert not output.exists()


def test_parser_state_is_restored_after_success_and_failure() -> None:
    before = getattr(argparse.ArgumentParser, "_check_help")  # noqa: B009
    utils = importlib.import_module("hydra._internal.utils")
    parser = cast(Callable[[], argparse.ArgumentParser], utils.get_args_parser)
    assert isinstance(parser(), argparse.ArgumentParser)
    assert getattr(argparse.ArgumentParser, "_check_help") is before  # noqa: B009

    def fail() -> argparse.ArgumentParser:
        raise RuntimeError("Injected parser failure")

    with pytest.raises(RuntimeError, match="Injected parser failure"):
        adapter._compatible_args_parser(fail)  # pyright: ignore[reportPrivateUsage]
    assert getattr(argparse.ArgumentParser, "_check_help") is before  # noqa: B009
    with pytest.raises(ValueError):
        argparse.ArgumentParser().add_argument("--bad", help=cast(str, object()))


def test_ordinary_bad_help_is_not_hidden_inside_wrapper() -> None:
    def malformed() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--bad", help=cast(str, object()))
        return parser

    with pytest.raises(ValueError):
        adapter._compatible_args_parser(malformed)  # pyright: ignore[reportPrivateUsage]


def test_bootstrap_is_idempotent() -> None:
    utils = importlib.import_module("hydra._internal.utils")
    main = importlib.import_module("hydra.main")
    before: Any = utils.get_args_parser
    adapter.prepare_hydra()
    assert utils.get_args_parser is main.get_args_parser is before


def test_unqualified_hydra_version_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter, "_parser_patched", False)

    def other_version(name: str) -> str:
        return "0.0.0"

    monkeypatch.setattr(adapter, "version", other_version)
    with pytest.raises(RuntimeError, match=r"qualified hydra-core==1\.3\.6"):
        adapter.prepare_hydra()


def test_pinned_upstream_parser_still_needs_the_backport() -> None:
    utils = importlib.import_module("hydra._internal.utils")
    upstream = utils.get_args_parser.__wrapped__
    with pytest.raises(ValueError, match="help"):
        upstream()


def test_base_package_does_not_require_hydra() -> None:
    code = """
import sys
from importlib.abc import MetaPathFinder
class BlockHydra(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'hydra', 'omegaconf'}:
            raise ImportError('Optional dependency intentionally absent')
sys.meta_path.insert(0, BlockHydra())
import marl_battlegrounds
from marl_battlegrounds import experiments
from marl_battlegrounds.training import TrainConfig
TrainConfig()
try:
    from marl_battlegrounds.experiments import hydra
except ImportError as error:
    assert 'marl-battlegrounds[hydra]' in str(error)
else:
    raise AssertionError('Optional dependency was required')
assert 'jax' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("operation", ["fresh", "resume", "extend"])
def test_missing_reports_fail_before_training_setup(
    tmp_path: Path, operation: str
) -> None:
    program = r"""import sys
from importlib.abc import MetaPathFinder
from pathlib import Path
class MissingPlots(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'matplotlib':
            raise ModuleNotFoundError('Injected missing report dependency')
sys.meta_path.insert(0, MissingPlots())
from marl_battlegrounds.training import runner
from marl_battlegrounds.training import TrainConfig

def forbidden(*args, **kwargs):
    raise AssertionError('Reached training setup or output creation')
runner._bind_training_population = forbidden
Path.mkdir = forbidden
output = Path(sys.argv[1]) / 'run'
operation = sys.argv[2]
try:
    if operation == 'fresh':
        runner.train(TrainConfig(), output_dir=output)
    elif operation == 'resume':
        runner.train(resume_from=output / 'checkpoints' / 'missing')
    else:
        runner.extend_training(output / 'checkpoints' / 'missing',
                               additional_env_steps=32, output_dir=output)
except ImportError as error:
    assert 'marl-battlegrounds[training]' in str(error)
else:
    raise AssertionError('Missing reports did not stop training')
assert not output.exists()
assert 'jax' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(tmp_path), operation],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
