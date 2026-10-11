"""Check the analysis notebook: its saved run, and a run on your own tournament.

Contracts: the saved notebook holds the outputs of one complete run, with all six
charts and no local folder in any output (the run order and error checks for both
notebooks live in ``tests/cli/test_install_start_script.py``). With
``MARL_BGS_RESULTS`` naming a small saved tournament of its own (three entrants
on one map, full metrics), the notebook runs from top to bottom in another
directory, draws the same six charts, never imports JAX, Flax or Optax, and
leaves the saved tournament and the notebook unchanged. The second check needs
the optional notebook extra and skips without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import marl_battlegrounds as marl_bgs

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "examples/evaluate/evaluation_analysis.ipynb"
CHARTS = 6


def _hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _pictures(notebook: dict[str, Any]) -> int:
    return sum(
        "image/png" in output.get("data", {})
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for output in cell["outputs"]
    )


def test_saved_run_shows_every_chart_and_no_local_folder() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert _pictures(json.loads(text)) == CHARTS
    for folder in ("/home/", "/tmp/", "/Users/", "C:\\\\"):
        assert folder not in text


def test_notebook_runs_on_your_own_tournament_without_jax(tmp_path: Path) -> None:
    pytest.importorskip("nbconvert", reason="Install the optional notebook extra")
    pytest.importorskip("ipykernel", reason="Install the optional notebook extra")
    pytest.importorskip("pandas", reason="Install the optional notebook extra")
    saved = marl_bgs.run_tournament(
        ["tdm-alpha", "tdm-beta", "random"],
        maps=[47],
        episodes_per_pair=2,
        max_steps=150,
        num_envs=2,
        metrics="full",
        output_dir=tmp_path / "tournament",
    ).run_dir
    assert saved is not None
    before = _hashes(saved)
    notebook_hash = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    work = tmp_path / "different-directory"
    work.mkdir()
    copied = work / NOTEBOOK.name
    shutil.copy2(NOTEBOOK, copied)
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text("""import importlib.abc, sys
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'flax', 'optax'}:
            raise AssertionError('Notebook imported training backend: ' + fullname)
sys.meta_path.insert(0, NoTraining())
""")
    kernel = tmp_path / "kernel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            str(kernel),
            "--name",
            "marl-bgs-notebook-test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cache = tmp_path / "cache"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MARL_BGS_")
    }
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": os.pathsep.join((str(guard), str(REPO / "src"))),
            "MARL_BGS_RESULTS": str(saved),
            "JUPYTER_PATH": str(kernel / "share/jupyter"),
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            "IPYTHONDIR": str(cache / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(cache / "jupyter"),
        }
    )
    executed = tmp_path / "executed.ipynb"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(copied),
            "--output",
            str(executed),
            "--ExecutePreprocessor.timeout=-1",
            "--ExecutePreprocessor.kernel_name=marl-bgs-notebook-test",
        ],
        cwd=work,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _hashes(saved) == before
    assert hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() == notebook_hash
    assert _pictures(json.loads(executed.read_text(encoding="utf-8"))) == CHARTS
