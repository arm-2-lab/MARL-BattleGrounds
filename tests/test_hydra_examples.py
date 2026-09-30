"""Check all tutorial launchers through real Hydra subprocesses without games.

Fresh-process guards prove help/config are host-only. Counted lesson bodies prove
composition and dispatch, not numerical behavior; existing workflow tests own
that behavior. Shared failures, paths and resources use this same adapter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.dev.generate_experiment_configs import _example_presets

# pyright: reportPrivateUsage=false

EXAMPLES = tuple(Path(name).stem for name in _example_presets())
REPO = Path(__file__).resolve().parents[1]
GUARD = """
import importlib.abc
import sys
class NoNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'flax', 'optax'}:
            raise AssertionError('Numerical import: ' + fullname)
sys.meta_path.insert(0, NoNumerics())
"""


def _process(
    tmp_path: Path, program: str, arguments: list[str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-c", program, *arguments],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join((str(REPO), str(REPO / "src"))),
        },
    )


def _spy(name: str, *, result: int = 0, failure: bool = False) -> str:
    return (
        GUARD
        + f"""
import functools
import importlib
import json
import os
from pathlib import Path
module = importlib.import_module('examples.{name}')
entry = {"launch" if name == "cross_play_and_zsc" else "run"!r}
original = getattr(module, entry)
@functools.wraps(original)
def lesson(**settings):
    print('DISPATCH ' + json.dumps({{'settings': settings, 'cwd': str(Path.cwd()),
        'platform': os.environ.get('JAX_PLATFORMS')}}, default=str))
    if {failure!r}:
        raise RuntimeError('Declared lesson failure')
    return {result}
setattr(module, entry, lesson)
# The collection lesson's display callback expects its real mapping return.
if hasattr(module, '_print_summary'):
    module._print_summary = lambda value: None
raise SystemExit(module.main())
"""
    )


def _required(name: str) -> list[str]:
    values = {
        "checkpoint": "saved/checkpoint",
        "output_dir": "chosen",
        "run_dirs": "[saved/run]",
        "declaration": "choice.json",
        "model": "host-only-fixture",
        "additional_env_steps": "4",
    }
    preset = _example_presets()[f"examples/{name}.yaml"]
    return [
        f"{field}={values[field]}" for field, value in preset.items() if value == "???"
    ]


@pytest.mark.parametrize("name", EXAMPLES)
@pytest.mark.parametrize("option", ["--help", "--cfg"])
def test_all_tutorial_help_and_config_skip_numerical_imports(
    tmp_path: Path, name: str, option: str
) -> None:
    program = (
        GUARD
        + f"""
import runpy
runpy.run_path({str(REPO / "examples")!r} + '/{name}.py', run_name='__main__')
"""
    )
    result = _process(
        tmp_path, program, [option, "job"] if option == "--cfg" else [option]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "preview" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_hydra_main_dispatches_its_existing_lesson_once(
    tmp_path: Path, name: str
) -> None:
    result = _process(tmp_path, _spy(name), _required(name))
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [
        line.removeprefix("DISPATCH ")
        for line in result.stdout.splitlines()
        if line.startswith("DISPATCH ")
    ]
    assert len(rows) == 1
    assert json.loads(rows[0])["cwd"] == str(tmp_path)
    assert "Tutorial launch records:" in result.stdout


@pytest.mark.parametrize(
    "name,method",
    [
        ("partner_training", method)
        for method in ("mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn")
    ]
    + [
        ("mappo_training", method)
        for method in ("mappo", "ippo", "ff_mappo", "ff_ippo")
    ],
)
def test_lesson_method_override_is_a_plain_value(
    tmp_path: Path, name: str, method: str
) -> None:
    result = _process(
        tmp_path,
        _spy(name),
        ["output_dir=chosen", f"method={method}", "resources=cpu"],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    row = next(
        line.removeprefix("DISPATCH ")
        for line in result.stdout.splitlines()
        if line.startswith("DISPATCH ")
    )
    dispatched = json.loads(row)
    assert dispatched["settings"]["method"] == method
    assert dispatched["platform"] == "cpu"
    assert not {"ppo", "qmix", "pqn", "recipe"}.intersection(dispatched["settings"])


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["invented=3"], "invented"),
        (["+invented=3"], "unexpected keyword"),
        (["-m", "seed=1,2"], "study"),
    ],
)
def test_bad_settings_fail_before_lesson_body(
    tmp_path: Path, arguments: list[str], message: str
) -> None:
    result = _process(tmp_path, _spy("training_distributions"), arguments)
    assert result.returncode != 0
    assert "DISPATCH " not in result.stdout
    assert message.lower() in (result.stdout + result.stderr).lower()


def test_preview_paths_resources_and_plain_values(tmp_path: Path) -> None:
    location = tmp_path / "chosen path"
    arguments = [
        f"output_dir={location}",
        "num_envs=4",
        "mappo=true",
        "resources=cpu",
        "preview=true",
    ]
    result = _process(tmp_path, _spy("training_distributions"), arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DISPATCH " not in result.stdout and "Input identities" in result.stdout
    assert not location.exists()
    arguments[-1] = "hydra.job.chdir=true"
    result = _process(tmp_path, _spy("training_distributions"), arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    row = next(
        line.removeprefix("DISPATCH ")
        for line in result.stdout.splitlines()
        if line.startswith("DISPATCH ")
    )
    data = json.loads(row)
    assert data["cwd"] == str(tmp_path) and data["platform"] == "cpu"
    assert data["settings"]["output_dir"] == str(location)
    assert data["settings"]["num_envs"] == 4 and data["settings"]["mappo"] is True
    assert not location.exists()


@pytest.mark.parametrize("failure,status", [(False, 1), (True, 1)])
def test_lesson_failure_keeps_exit_status_and_output_location(
    tmp_path: Path, failure: bool, status: int
) -> None:
    result = _process(
        tmp_path, _spy("training_distributions", result=status, failure=failure), []
    )
    assert result.returncode == status
    assert "Tutorial launch records:" in result.stdout + result.stderr
    if failure:
        assert "Declared lesson failure" in result.stderr


def test_launch_records_cannot_occupy_the_lesson_output(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = _process(
        tmp_path,
        _spy("training_distributions"),
        [
            f"output_dir={output}",
            f"hydra.run.dir={output}",
        ],
    )
    assert result.returncode != 0
    assert "outside the experiment destination" in result.stderr
    assert "DISPATCH " not in result.stdout and not output.exists()


@pytest.mark.parametrize(
    "name,flag",
    [
        ("mappo_training", "evaluate"),
        ("qmix_training", "evaluate"),
        ("pqn_training", "curriculum"),
        ("pqn_training", "shaping"),
        ("pqn_training", "evaluate"),
        ("own_learner", "custom_reward"),
        ("training_distributions", "mappo"),
        ("training_study", "background"),
        ("canonical_tournament", "save"),
        ("canonical_tournament", "prepare"),
        ("canonical_tournament", "rerun_existing"),
    ],
)
def test_plain_lesson_rejects_text_flags_before_work(
    tmp_path: Path, name: str, flag: str
) -> None:
    program = (
        GUARD
        + f"""
import importlib
import inspect
module = importlib.import_module('examples.{name}')
parameters = inspect.signature(module.run).parameters
settings = {{key: 'unused' for key, value in parameters.items()
    if value.default is inspect.Parameter.empty}}
settings[{flag!r}] = 'false'
try:
    module.run(**settings)
except TypeError as error:
    assert '{flag} must be true' in str(error), str(error)
else:
    raise AssertionError('Text flag accepted')
"""
    )
    result = _process(tmp_path, program, [])
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("flag", ["prepare", "save", "rerun_existing"])
def test_hydra_rejects_quoted_boolean_without_fixture_or_download(
    tmp_path: Path, flag: str
) -> None:
    program = (
        GUARD
        + """
from examples.canonical_tournament import main
raise SystemExit(main())
"""
    )
    result = _process(tmp_path, program, [f"{flag}='false'"])
    assert result.returncode != 0
    assert f"{flag} must be true" in result.stderr
    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize("field", ["directory", "output_dir"])
def test_canonical_destinations_stay_outside_hydra_records(
    tmp_path: Path, field: str
) -> None:
    output = tmp_path / "chosen"
    result = _process(
        tmp_path,
        _spy("canonical_tournament"),
        [f"{field}={output}", f"hydra.run.dir={output}"],
    )
    assert result.returncode != 0
    assert "outside the experiment destination" in result.stderr
    assert "DISPATCH " not in result.stdout and not output.exists()


@pytest.mark.parametrize("entry", ["import", "module", "script"])
def test_canonical_fixture_runs_as_callable_module_and_script(
    tmp_path: Path, entry: str
) -> None:
    fixture_path = REPO / "examples/canonical_tournament.py"
    if entry == "import":
        program = (
            ""
            + """
from examples.canonical_tournament import run
run(command='fixture', entrants=2)
"""
        )
        arguments = []
    else:
        program = (
            ""
            + "\nimport runpy\n"
            + (
                "runpy.run_module('examples.canonical_tournament', run_name='__main__')"
                if entry == "module"
                else (f"runpy.run_path({str(fixture_path)!r}, run_name='__main__')")
            )
        )
        # A directly launched file has its own folder on sys.path.
        if entry == "script":
            program = program.replace(
                "import runpy",
                f"import runpy, sys\nsys.path.insert(0, {str(REPO / 'examples')!r})",
            )
        arguments = ["entrants=2"]
    result = _process(tmp_path, program, arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Rankings:" in result.stdout


@pytest.mark.parametrize("field", ["directory", "output_dir"])
def test_canonical_launch_records_default_to_chosen_destination_sibling(
    tmp_path: Path, field: str
) -> None:
    result = _process(
        tmp_path, _spy("canonical_tournament"), [f"{field}=chosen", "preview=true"]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(tmp_path / "chosen.launches") in result.stdout
    assert not (tmp_path / "chosen").exists()


def test_public_depth_check_keeps_the_host_only_shared_rule(tmp_path: Path) -> None:
    program = (
        GUARD
        + """
from marl_battlegrounds.training.config import (
    validate_red_zone_depth, _validate_red_zone_depth)
assert validate_red_zone_depth is _validate_red_zone_depth
assert validate_red_zone_depth(0.0) == 0.0
assert validate_red_zone_depth(5.0) == 5.0
for value in (True, 1, float('nan'), float('inf'), -0.0, -1.0, 1e300, 1e-300):
    try:
        validate_red_zone_depth(value)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError(value)
"""
    )
    result = _process(tmp_path, program, [])
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("target", ["result", "launch", "records", "cache"])
def test_scenario_hydra_protects_actor_inputs_before_writes(
    tmp_path: Path, target: str
) -> None:
    actor = tmp_path / "actor"
    actor.mkdir()
    (actor / "actor_details.json").write_text("Do not read or alter this input")
    options = [f"system={actor}", f"output_dir={tmp_path / 'result'}"]
    options.extend(
        {
            "result": [f"output_dir={actor / 'results'}"],
            "launch": [f"hydra.run.dir={actor / 'launch'}"],
            "records": [f"hydra.output_subdir={actor / 'records'}"],
            "cache": [
                "resources=cpu",
                f"+resources.compilation_cache={actor / 'cache'}",
            ],
        }[target]
    )
    result = _process(tmp_path, _spy("scenario_evaluation"), options)
    assert result.returncode != 0
    assert "DISPATCH " not in result.stdout
    assert "outside input artifacts" in result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["actor"]


def test_scenario_resume_launch_records_are_siblings(tmp_path: Path) -> None:
    saved = tmp_path / "saved-result"
    saved.mkdir()
    result = _process(tmp_path, _spy("scenario_evaluation"), [f"resume_from={saved}"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "saved-result.launches").is_dir()
    assert list(saved.iterdir()) == []


@pytest.mark.parametrize("linked", [False, True])
def test_scenario_hydra_preflights_actual_child_before_launch_records(
    tmp_path: Path, linked: bool
) -> None:
    output = tmp_path / "chosen"
    output.mkdir()
    actor = tmp_path / "actor" if linked else output / "scenario-1"
    actor.mkdir()
    (actor / "actor_details.json").write_text("No model load")
    if linked:
        (output / "scenario-1").symlink_to(actor, target_is_directory=True)
    result = _process(
        tmp_path,
        _spy("scenario_evaluation"),
        [f"system={actor}", f"output_dir={output}"],
    )
    assert result.returncode != 0 and "DISPATCH " not in result.stdout
    assert "outside input artifacts" in result.stderr or "symbolic" in result.stderr
    assert not (tmp_path / "chosen.launches").exists()
    assert list(actor.iterdir()) == [actor / "actor_details.json"]
