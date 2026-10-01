"""Prepare explicit software workflows through Hydra and the existing API.

analysis_software_check copies labelled synthetic saved results. software_check
verifies the frozen opponent pool and writes a real six-method study declaration.
Neither route launches training or games; actual paper settings come later.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

# Import the optional adapter first for its clear installation error.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra, resolve_training_config

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
# isort: on


def prepare_analysis_software_check(output_dir: str | Path) -> Path:
    """Copy synthetic saved rows and return their complete grouping YAML path.

    output_dir names the chosen example root. Its calibration and protocol
    subfolders must not exist; unrelated sibling files, including Hydra records,
    are preserved. No existing experiment is overwritten. Read installed fixture
    resources, copy them and write a declaration with file-relative run paths.
    Returns an absolute path. Missing resources or collisions raise OSError.
    """
    root = Path(output_dir).expanduser().resolve()
    if any((root / name).exists() for name in ("calibration", "protocol", "README.md")):
        raise FileExistsError(
            "Choose an output directory without calibration, protocol "
            "or README.md outputs"
        )
    source = files("marl_battlegrounds.experiments").joinpath(
        "fixtures/recipe_comparison"
    )
    # Installed wheels contain ordinary resources. Keep copying separate from
    # training APIs: these are labelled saved fixtures, not new experiment runs.
    shutil.copytree(str(source.joinpath("calibration")), root / "calibration")
    target = root / "protocol/grouping.yaml"
    target.parent.mkdir(parents=True)
    declaration = json.loads(source.joinpath("grouping.json").read_text())
    OmegaConf.save(OmegaConf.create(declaration), target)
    (root / "README.md").write_text(source.joinpath("README.md").read_text())
    return target


def prepare_software_check(opponent_bundle: str | Path, output_dir: str | Path) -> Path:
    """Prepare the six-method wiring check from an explicit frozen opponent pool.

    opponent_bundle names a local folder containing pool.json and its declared
    actors. Verify the selected eight identities and three scripted names before
    writing. output_dir receives protocol/study.yaml, checkpoint-selection.yaml,
    grouping.yaml, design.csv and source.json. Paths are relative to protocol/;
    move the input bundle with the workflow to keep them valid. A matching
    prepared protocol is reusable; conflicting outputs are refused. Return the
    study YAML path. Keep outputs outside the bundle; symbolic path aliases are
    refused before any write. Load only the validator to obtain its panel identity;
    run no games or learning. Use JAX_PLATFORMS=cpu for this preparation command.
    Missing/corrupt inputs raise before publication. These fixed small settings
    test software and are never presented as paper hyperparameters. The four
    PPO methods use combat_support_reward; QMIX and PQN-VDN, which train on one
    team reward, use its team form combat_support_team_reward.
    """
    from marl_battlegrounds.experiments.hydra import (
        _preparation_paths,  # pyright: ignore[reportPrivateUsage]
    )

    bundle, root = _preparation_paths(opponent_bundle, output_dir)
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.methods import TRAINING_METHODS
    from marl_battlegrounds.training._checkpoint_metadata import (
        _directory,  # pyright: ignore[reportPrivateUsage]
        _read_json,  # pyright: ignore[reportPrivateUsage]
        _relative,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training.checkpoints import (
        artifact_identity,
        runtime_identity,
    )
    from marl_battlegrounds.training.validation import VALIDATION_MAPS

    bundle = _directory(bundle).resolve()
    source = bundle / "pool.json"
    raw = source.read_bytes()
    pool = _read_json(source)
    expected = json.loads(
        files("marl_battlegrounds.experiments")
        .joinpath("fixtures/combat_support_opponents.json")
        .read_text()
    )
    if (
        type(pool.get("schema_version")) is not int
        or pool["schema_version"] != 1
        or pool.get("validation_member") != expected["validation_member"]
    ):
        raise ValueError("Opponent bundle has a different schema or validator")
    members: object = pool.get("members")
    if not isinstance(members, list):
        raise ValueError("Opponent bundle members must be a list")
    members = cast(list[Any], members)
    if len(members) != len(expected["members"]):
        raise ValueError(
            "Software check requires the declared eight actors and three scripts"
        )
    references: dict[str, str] = {}
    for member, frozen in zip(members, expected["members"], strict=True):
        if not isinstance(member, dict):
            raise ValueError(f"Opponent member must be an object: {frozen['name']}")
        member = cast(dict[str, Any], member)
        if any(member.get(key) != value for key, value in frozen.items()):
            raise ValueError(f"Frozen opponent identity differs: {frozen['name']}")
        name = member["name"]
        if member["kind"] == "scripted":
            if member.get("reference", name) != name:
                raise ValueError(f"Scripted opponent reference differs: {name}")
            references[name] = name
            continue
        if not isinstance(member.get("path"), str) or not member["path"]:
            raise ValueError(f"Opponent needs a relative actor path: {name}")
        actor = _relative(bundle, member["path"])
        identity = artifact_identity(actor)
        if (
            identity["checkpoint_id"] != frozen["actor_export_id"]
            or identity["actor_digest"] != frozen["inference_digest"]
            or identity["metadata"].get("checkpoint_id")
            != frozen["source_checkpoint_id"]
        ):
            raise ValueError(f"Opponent bytes differ from the frozen identity: {name}")
        references[name] = str(actor.resolve())
    destination = root / "protocol"
    validator = references[expected["validation_member"]]
    # The shared panel owner determines labels and identity. Its path is not part
    # of the digest; the runner independently creates the same declared panel.
    with tempfile.TemporaryDirectory(prefix="marl-software-panel-") as temporary:
        panel = training.create_panel(opponents=[validator], output_dir=temporary)
    relative = {
        name: os.path.relpath(value, destination)
        if name not in ("tdm-alpha", "tdm-beta", "tdm-gamma")
        else value
        for name, value in references.items()
    }
    order = ("mappo", "ff_mappo", "ippo", "ff_ippo", "qmix", "pqn_vdn")
    if set(order) != set(TRAINING_METHODS):
        raise ValueError("Software-check method census differs from training methods")
    cases: list[dict[str, Any]] = []
    run_rows: list[dict[str, str]] = []
    design = io.StringIO()
    writer = csv.DictWriter(design, fieldnames=("case_id", "method", "seed", "purpose"))
    writer.writeheader()
    for method in order:
        recipe = (
            files("marl_battlegrounds.experiments")
            .joinpath(f"configs/recipe_settings/software_check_{method}.yaml")
            .read_text()
        )
        config = cast(
            dict[str, Any],
            OmegaConf.to_container(OmegaConf.create(recipe), resolve=True),
        )
        seed = config.pop("seed")
        config.update(
            method=method,
            reward="marl_battlegrounds.training.reward_recipes:"
            + (
                "combat_support_team_reward"
                if method in ("qmix", "pqn_vdn")
                else "combat_support_reward"
            ),
            opponents={
                "self": 0.6,
                "past": 0.2,
                **{name: 0.2 / 11 for name in references},
            },
            validation_opponents=[relative[expected["validation_member"]]],
        )
        resolve_training_config({**config, "seed": seed}, base_dir=destination)
        identifier = f"{method}-software-check"
        cases.append(
            {
                "id": identifier,
                "config": config,
                "seeds": [seed],
                "duration_seconds": 900,
                "runtime": {"opponents": relative},
            }
        )
        full_id = f"{identifier}--seed-{seed}"
        run_rows.append(
            {
                "run_dir": f"../tuning/study/cases/{full_id}/run",
                "group": method,
                "unit": f"seed-{seed}",
            }
        )
        writer.writerow(
            {
                "case_id": full_id,
                "method": method,
                "seed": seed,
                "purpose": "Software Check",
            }
        )
    study = {
        "schema_version": 1,
        "name": "Six Method Software Check",
        "failure_policy": "stop",
        "cases": cases,
    }
    grouping = {
        "schema_version": 2,
        "measure": "point_margin",
        "maps": list(VALIDATION_MAPS),
        "opponents": [member.name for member in panel.members],
        "panel_digest": panel.digest,
        "red_zone_depth": 5.0,
        "runs": run_rows,
        "comparisons": [],
    }
    selection = {
        "schema_version": 1,
        "name": "Software Check Point Margin",
        "rule": "point_margin",
        "confirmation_seed_pairs": 1,
    }
    records = {
        "study.yaml": OmegaConf.to_yaml(OmegaConf.create(study)),
        "grouping.yaml": OmegaConf.to_yaml(OmegaConf.create(grouping)),
        "checkpoint-selection.yaml": OmegaConf.to_yaml(OmegaConf.create(selection)),
        "design.csv": design.getvalue(),
        "source.json": json.dumps(
            {
                "schema_version": 1,
                "meaning": (
                    "Software qualification only; "
                    "no learning claim or paper search settings"
                ),
                "opponent_pool": {
                    "path": os.path.relpath(source, destination),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "validation_member": expected["validation_member"],
                "members": expected["members"],
                "settings_source": "Packaged software_check recipes",
                "runtime_identities": {
                    method: runtime_identity(method) for method in order
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    }
    if source.read_bytes() != raw:
        raise ValueError("Opponent declaration changed during preparation")
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_dir()
            or set(path.name for path in destination.iterdir()) != set(records)
        ):
            raise ValueError(f"Prepared protocol conflicts: {destination}")
        if any(
            (destination / name).is_symlink()
            or (destination / name).read_text() != text
            for name, text in records.items()
        ):
            raise ValueError(f"Prepared protocol conflicts: {destination}")
        return destination / "study.yaml"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".protocol-", dir=root) as temporary:
        staged = Path(temporary) / "protocol"
        staged.mkdir()
        for name, text in records.items():
            (staged / name).write_text(text)
        staged.rename(destination)
    return destination / "study.yaml"


@hydra.main(version_base="1.3", config_path="configs", config_name="prepare")
def main(config: DictConfig) -> Path:
    """Prepare the software example and return/print its exact declaration path.

    analysis_software_check copies synthetic saved rows. software_check verifies
    an explicit opponent_bundle and prepares the six-method declaration. Neither
    launches a study. Relative command paths use the original working directory.
    Unknown protocols or missing inputs fail before protocol output is written.
    """
    output = to_absolute_path(str(Path(config.output_dir).expanduser()))
    if config.protocol == "analysis_software_check":
        result = prepare_analysis_software_check(output)
    elif config.protocol == "software_check":
        if not config.opponent_bundle:
            raise ValueError(
                "software_check needs opponent_bundle=/path/frozen-opponents"
            )
        result = prepare_software_check(
            to_absolute_path(str(Path(config.opponent_bundle).expanduser())), output
        )
    else:
        raise ValueError("Choose protocol=software_check or analysis_software_check")
    print(result)
    return result


if __name__ == "__main__":
    prepare_hydra()
    main()
