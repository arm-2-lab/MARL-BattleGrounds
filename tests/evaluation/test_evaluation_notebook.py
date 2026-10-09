"""Execute the saved-results notebook without simulation or changing raw inputs.

The optional notebook extra supplies the real kernel and nbconvert runner. Tests
copy the actual notebook to another directory, retain source hashes and block
training backend imports. Ordinary source checks still run without that extra.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "examples/evaluation_analysis.ipynb"
BUNDLE = REPO / "src/marl_battlegrounds/examples_data/evaluation"


def _hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_notebook_is_unexecuted_and_demo_is_small() -> None:
    data = json.loads(NOTEBOOK.read_text())
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
    assert (
        sum(path.stat().st_size for path in BUNDLE.rglob("*") if path.is_file())
        < 5 * 1024**2
    )
    assert (BUNDLE / "run_details.json").is_file()
    assert len(list((BUNDLE / "replays").glob("*.json"))) == 2


@pytest.mark.parametrize("case", ["demo", "missing", "overlap", "nonempty"])
def test_actual_notebook_reads_or_reports_invalid_paths_without_training(
    tmp_path: Path, case: str
) -> None:
    pytest.importorskip("nbconvert", reason="Install the optional notebook extra")
    pytest.importorskip("ipykernel", reason="Install the optional notebook extra")
    pytest.importorskip("pandas", reason="Install the optional notebook extra")
    before = _hashes(BUNDLE)
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
    report = tmp_path / "report"
    report.mkdir()
    if case == "nonempty":
        (report / "selected_metrics.csv").write_text("Existing report data\n")
    report_before = _hashes(report)
    cache = tmp_path / "cache"
    cache.mkdir()
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
            "MARL_BGS_REPORT_DIR": str(report),
            "JUPYTER_PATH": str(kernel / "share/jupyter"),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            "IPYTHONDIR": str(cache / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(cache / "jupyter"),
        }
    )
    if case == "missing":
        env["MARL_BGS_RESULTS"] = str(tmp_path / "missing")
    elif case == "overlap":
        env["MARL_BGS_REPORT_DIR"] = str(BUNDLE / "forbidden")
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
            str(report / "executed.ipynb"),
            "--ExecutePreprocessor.timeout=-1",
            "--ExecutePreprocessor.kernel_name=marl-bgs-notebook-test",
        ],
        cwd=work,
        env=env,
        text=True,
        capture_output=True,
    )
    assert _hashes(BUNDLE) == before
    assert hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() == notebook_hash
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == notebook_hash
    if case == "demo":
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((report / "analysis_preview.json").read_text())
        import marl_battlegrounds as marl_bgs

        saved = marl_bgs.load_results(BUNDLE)
        for name, facts in saved.metadata["tables"].items():
            for key in (
                "missing_historical_fields",
                "unavailable_columns",
                "missing_passes",
            ):
                assert key in summary["coverage"]["tables"][name]
                assert summary["coverage"]["tables"][name][key] == json.loads(
                    json.dumps(facts.get(key))
                )
        assert summary["software_demo"] is True
        assert Path(summary["kernel_directory"]) == work
        assert len(summary["game_preview"]) == 2
        assert len(summary["metric_preview"]) == 1
        assert len(summary["replays"]) == 2
        assert summary["point_margins"][0]["ci_low"] is None
        assert summary["point_margins"][0]["unavailable_reason"]
        assert (report / "point_margins.png").is_file()
        from marl_battlegrounds.evaluation.metric_catalog import METRIC_TOPICS

        assert {row["topic"] for row in summary["topic_coverage"]} == {
            topic.name for topic in METRIC_TOPICS
        }
        assert {
            row["scope"] for row in summary["metric_groups"] if row["finite_values"]
        } == {
            "episode",
            "team",
            "agent",
            "team_recipient",
            "source_recipient",
            "ally_pair",
        }
        with (report / "selected_metrics.csv").open() as stream:
            values = {row["name"]: row["value"] for row in csv.DictReader(stream)}
        assert float(values["agent_4_to_agent_3_healing_done"]) == 8
        assert float(values["team_b_healing_done"]) == 208
        assert float(values["team_a_rescue_opportunities"]) == 0
        assert values["team_a_rescue_rate"] == ""
        assert summary["behavior_sample"]["examined_games"] == 1
        selected = [
            row for row in summary["replay_index"] if row["selected_detail_game"]
        ]
        assert len(selected) == 1
        assert selected[0]["config_id"] == summary["detail_game"]["config_id"]
        assert str(selected[0]["episode_id"]) == str(
            summary["detail_game"]["episode_id"]
        )
        for chart in summary["charts"]:
            assert (report / chart["png"]).is_file()
            assert (report / chart["svg"]).is_file()
            assert chart["data"] and (report / chart["data"]).is_file()
        with (report / "relationships.csv").open() as stream:
            relationships = list(csv.DictReader(stream))
        team_recipients = [
            row for row in relationships if row["scope"] == "team_recipient"
        ]
        assert team_recipients
        assert {row["stem"] for row in team_recipients} == {"basic_applications"}
        assert {row["unit"] for row in team_recipients} == {"count"}
        assert all(row["value"] == values[row["name"]] for row in team_recipients)
        executed = json.loads((report / "executed.ipynb").read_text())
        pictures = [
            output
            for cell in executed["cells"]
            if cell["cell_type"] == "code"
            for output in cell["outputs"]
            if "image/png" in output.get("data", {})
        ]
        assert len(pictures) == len(summary["charts"])
    else:
        assert result.returncode != 0
        message = {
            "missing": "exact saved result folder",
            "overlap": "outside the saved input",
            "nonempty": "new or empty report folder",
        }[case]
        assert message in result.stdout + result.stderr
        assert not (report / "analysis_preview.json").exists()
        assert _hashes(report) == report_before
