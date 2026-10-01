"""Check that saved validation games still verify after the source code changes.

A validation pass records the code revision that played it. Continuing a run
on newer code re-checks that saved pass read-only; the check keeps the pass's
own recorded revision, so the saved games are accepted. Ordinary pass identity
for new writing still rejects a changed code revision, and a changed scientific
seed is still refused during the read-only check.
"""

# pyright: reportPrivateUsage=false
import json
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.evaluation.policy_execution import Policy, policy
from marl_battlegrounds.evaluation.run_writer import _prepare_pass_identity
from marl_battlegrounds.training import checkpoints, validation


def test_saved_pass_verifies_under_changed_source_but_new_writes_do_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = validation.create_panel(
        opponents=("tdm-alpha",), output_dir=tmp_path / "panel"
    )

    def artifact(path: str | Path) -> dict[str, Any]:
        return {"checkpoint_id": "checked", "actor_digest": "actor", "env_steps": 32}

    def load_system(path: str | Path) -> Policy:
        return policy("tdm-beta")

    monkeypatch.setattr(validation, "_artifact", artifact)
    monkeypatch.setattr(checkpoints, "load_system", load_system)
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    evaluate, verify = evaluator.evaluate, evaluator._verify_evaluation
    saved = evaluator.capture_recording_provenance(num_envs=2)
    current = deepcopy(saved)
    current["code_revision"] = {
        **current["code_revision"],
        "source_tree_digest": "0" * 64,
    }
    provenance = {"value": saved}

    def pinned_provenance(*, num_envs: int) -> dict[str, Any]:
        return deepcopy(provenance["value"])

    def short(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return evaluate(*args, max_steps=1, **kwargs)

    def short_verify(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return verify(*args, max_steps=1, **kwargs)

    monkeypatch.setattr(evaluator, "capture_recording_provenance", pinned_provenance)
    monkeypatch.setattr(evaluator, "evaluate", short)
    monkeypatch.setattr(evaluator, "_verify_evaluation", short_verify)
    folder = tmp_path / "task"
    result = validation.validate_checkpoint(
        tmp_path / "actor", panel, output_dir=folder, seed_pairs=1, num_envs=2
    )
    run_dir = Path(result["pass_paths"][0])
    task = json.loads((folder / "task.json").read_text())
    manifest_path = run_dir / "run_details.json"
    before = manifest_path.read_bytes()

    provenance["value"] = current
    actor, opponent = policy("tdm-beta"), policy("tdm-alpha")
    validation._verify_panel_pass(
        run_dir, actor, opponent, task=task, member_index=0, num_envs=2
    )
    changed_seed = deepcopy(task)
    changed_seed["members"][0]["root"] += 1
    with pytest.raises(ValueError, match="seed"):
        validation._verify_panel_pass(
            run_dir, actor, opponent, task=changed_seed, member_index=0, num_envs=2
        )
    assert manifest_path.read_bytes() == before

    manifest = json.loads(before)
    record = next(iter(manifest["passes"].values()))
    assert record["details"]["code_revision"] == saved["code_revision"]
    with pytest.raises(ValueError, match="pass identity differs"):
        _prepare_pass_identity(
            manifest,
            record["phase"],
            record["pass_id"],
            record["policies"],
            record["checkpoint_id"],
            {**record["details"], "code_revision": current["code_revision"]},
        )
