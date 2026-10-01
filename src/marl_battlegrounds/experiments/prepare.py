"""Prepare explicit study workflows through Hydra and the existing API.

Three routes write a protocol/ folder that later stages read:

- analysis_software_check copies labelled synthetic saved results.
- software_check verifies the frozen opponent pool and writes a six-method
  wiring-check declaration with small settings.
- declared_study verifies an opponent bundle against a committed list of
  expected identities, binds a committed study declaration's public opponent
  names to the checked model folders and writes its grouping, selection,
  design and source records.

Both opponent routes share one bundle check (_read_opponent_pool and
_verify_opponent_members) and one publication step (_publish_protocol). No
route launches training or plays games.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
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


def _read_opponent_pool(bundle: Path) -> tuple[Path, Path, bytes, dict[str, Any]]:
    """Read an opponent bundle's pool.json once, keeping its exact bytes.

    Parameters
    ----------
    bundle : Path
        Opponent bundle folder, already checked by _preparation_paths. It must
        exist and no part of its path may be a symbolic link.

    Returns
    -------
    tuple of (Path, Path, bytes, dict)
        The resolved bundle folder, its pool.json path, the file's bytes and
        the parsed JSON object. Callers compare the bytes again before they
        publish, so a pool changed during preparation is refused.

    Raises
    ------
    ValueError
        The folder is missing or linked, or pool.json is missing, linked, not a
        JSON object or repeats a key.

    Notes
    -----
    Host-only. Reads one small file; loads no model and writes nothing.
    """
    from marl_battlegrounds.training._checkpoint_metadata import (
        _directory,  # pyright: ignore[reportPrivateUsage]
        _read_json,  # pyright: ignore[reportPrivateUsage]
    )

    bundle = _directory(bundle).resolve()
    source = bundle / "pool.json"
    raw = source.read_bytes()
    return bundle, source, raw, _read_json(source)


def _verify_opponent_members(
    bundle: Path,
    pool: Mapping[str, Any],
    expected: Sequence[Mapping[str, Any]],
    *,
    count_message: str,
) -> dict[str, str]:
    """Check every bundle member, in order, against its expected identity.

    Parameters
    ----------
    bundle : Path
        Resolved bundle folder that owns every actor path in pool.
    pool : mapping
        Parsed pool.json. Its "members" list must hold one object per expected
        member, in the same order.
    expected : sequence of mappings
        One record per member. Each needs "name" and "kind" ("actor" or
        "scripted"). Every key in a record must equal the same key in the pool
        member. An actor record also needs "actor_export_id",
        "inference_digest" and "source_checkpoint_id"; when it has
        "env_steps", the saved actor's training transitions must match too.
    count_message : str
        Error text used when the member count differs.

    Returns
    -------
    dict of str to str
        Member name to reference. An actor maps to its absolute resolved
        folder; a scripted member maps to its own installed name.

    Raises
    ------
    ValueError
        members is not a list, has the wrong length, holds a non-object, a
        declared field differs, a scripted reference differs, an actor path is
        missing, absolute, leaves the bundle or uses a link, or the saved
        actor's bytes or identity differ from the expected record.
    OSError
        A saved actor file cannot be read.

    Notes
    -----
    Host-only. training.checkpoints.artifact_identity hashes every actor file;
    no array is restored and no file is written.
    """
    from marl_battlegrounds.training._checkpoint_metadata import (
        _relative,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training.checkpoints import artifact_identity

    members: object = pool.get("members")
    if not isinstance(members, list):
        raise ValueError("Opponent bundle members must be a list")
    members = cast(list[Any], members)
    if len(members) != len(expected):
        raise ValueError(count_message)
    references: dict[str, str] = {}
    for member, frozen in zip(members, expected, strict=True):
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
            or ("env_steps" in frozen and identity["env_steps"] != frozen["env_steps"])
        ):
            raise ValueError(f"Opponent bytes differ from the frozen identity: {name}")
        references[name] = str(actor.resolve())
    return references


def _publish_protocol(root: Path, records: Mapping[str, str]) -> Path:
    """Write root/protocol once, or reuse an identical earlier preparation.

    Parameters
    ----------
    root : Path
        Chosen workflow folder, already checked by _preparation_paths. It is
        created when missing; unrelated files in it are kept.
    records : mapping of str to str
        File name to exact text. It must include "study.yaml".

    Returns
    -------
    Path
        root/protocol/study.yaml.

    Raises
    ------
    ValueError
        root/protocol exists but is a link, not a folder, holds other file
        names, or any file is a link or has different text.

    Notes
    -----
    New files are written into a hidden temporary folder beside protocol/ and
    then renamed in one step, so a failed write never leaves a partial
    protocol. An identical existing protocol is left untouched.
    """
    destination = root / "protocol"
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
    team reward, use its team form combat_support_team_reward. The bundle check
    and publication are shared with prepare_declared_study.
    """
    from marl_battlegrounds.experiments.hydra import (
        _preparation_paths,  # pyright: ignore[reportPrivateUsage]
    )

    bundle, root = _preparation_paths(opponent_bundle, output_dir)
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.methods import TRAINING_METHODS
    from marl_battlegrounds.training.checkpoints import runtime_identity
    from marl_battlegrounds.training.validation import VALIDATION_MAPS

    bundle, source, raw, pool = _read_opponent_pool(bundle)
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
    references = _verify_opponent_members(
        bundle,
        pool,
        expected["members"],
        count_message=(
            "Software check requires the declared eight actors and three scripts"
        ),
    )
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
    return _publish_protocol(root, records)


_IDENTITY_FIELDS = ("actor_export_id", "inference_digest", "source_checkpoint_id")
_ROLES = ("training", "validation")


def _expected_opponents(path: Path) -> list[dict[str, Any]]:
    """Read and check the committed list of expected bundle members.

    Parameters
    ----------
    path : Path
        JSON or YAML file with schema_version 1, a nonempty name and a members
        list. Each member has a distinct nonempty "name", "kind" ("actor" or
        "scripted") and "roles", a nonempty list drawn from "training" (new
        training games play it) and "validation" (it is in the validation
        panel). An actor also has "env_steps" (a positive integer) and the
        three identity strings actor_export_id, inference_digest and
        source_checkpoint_id. A scripted member has no other fields.

    Returns
    -------
    list of dict
        The members in file order, as plain copies.

    Raises
    ------
    ValueError
        The file is missing or malformed, or a member breaks the rules above.
    ImportError
        A YAML file is used without the hydra extra.

    Notes
    -----
    Host-only. Reads one small file and writes nothing.
    """
    from marl_battlegrounds.training._launch import read_declaration

    declared = read_declaration(path)
    if (
        set(declared) != {"schema_version", "name", "members"}
        or declared["schema_version"] != 1
        or not isinstance(declared["name"], str)
        or not declared["name"].strip()
        or not isinstance(declared["members"], list)
        or not declared["members"]
    ):
        raise ValueError(
            f"Expected opponents need schema_version 1, name and members: {path}"
        )
    members: list[dict[str, Any]] = []
    for raw in cast(list[Any], declared["members"]):
        if not isinstance(raw, dict):
            raise ValueError("Each expected opponent must be an object")
        member = cast(dict[str, Any], raw)
        roles = member.get("roles")
        fields = (
            {"name", "kind", "roles", "env_steps", *_IDENTITY_FIELDS}
            if member.get("kind") == "actor"
            else {"name", "kind", "roles"}
        )
        if (
            member.get("kind") not in ("actor", "scripted")
            or set(member) != fields
            or not isinstance(member["name"], str)
            or not member["name"].strip()
            or not isinstance(roles, list)
            or not roles
            or len(set(cast(list[Any], roles))) != len(cast(list[Any], roles))
            or any(role not in _ROLES for role in cast(list[Any], roles))
            or (
                member["kind"] == "actor"
                and (
                    type(member["env_steps"]) is not int
                    or member["env_steps"] <= 0
                    or any(
                        not isinstance(member[key], str) or not member[key]
                        for key in _IDENTITY_FIELDS
                    )
                )
            )
        ):
            raise ValueError(f"Expected opponent is malformed: {member.get('name')}")
        if member["name"] in {row["name"] for row in members}:
            raise ValueError(f"Expected opponent is repeated: {member['name']}")
        members.append(dict(member))
    return members


def _same_setting(cases: Sequence[Mapping[str, Any]], name: str, default: Any) -> Any:  # noqa: ANN401
    """Return one setting that every case shares, or refuse a mixed study.

    cases are declared training cases; name is a TrainConfig field; default
    is used where a case omits it (the API default). Grouping and checkpoint
    selection compare runs on one panel, map list, depth and game count, so
    the values must agree. Raise ValueError when they differ.
    """
    values = {
        json.dumps(case["config"].get(name, default), sort_keys=True) for case in cases
    }
    if len(values) != 1:
        raise ValueError(f"Every case must declare the same {name}")
    return json.loads(values.pop())


def prepare_declared_study(
    declaration: str | Path,
    opponent_bundle: str | Path,
    output_dir: str | Path,
    *,
    expected_opponents: str | Path | None = None,
    recipe_selection: str | Path | None = None,
) -> Path:
    """Check an opponent bundle and bind a committed study declaration to it.

    The committed declaration names its fixed opponents by public name: the
    keys of each case's config.opponents (besides "self" and "past") and the
    list runtime.validation_opponents. This function checks every bundle
    member against the expected identities, then writes output_dir/protocol/
    with five files:

    - study.yaml: the declaration with every actor name bound to its checked
      folder, as a path relative to protocol/ (runtime.opponents and
      runtime.validation_opponents). Scripted names stay names. Nothing else
      changes.
    - grouping.yaml: schema 2, point margin, the shared validation maps, the
      panel's opponent labels and digest, one group per case with one unit per
      seed, and run folders ../study/cases/<case>--seed-<seed>/run. A
      recipe_selection file adds its declared recipe picks.
    - checkpoint-selection.yaml: rule point_margin with the shared
      confirmation_seed_pairs.
    - design.csv: one row per run (case_id, recipe, method, seed).
    - source.json: SHA-256 hashes of the declaration, expected opponents,
      pool.json and recipe picks, the checked members and the runtime source
      identity of each method.

    Parameters
    ----------
    declaration : str or Path
        Committed JSON or YAML study declaration (training.run_study format).
        Every case must be a training case. It must not bind runtime.opponents
        itself, and must not use config.validation_opponents or
        config.validation_panel. Its config.opponents names must be exactly the
        expected members with the "training" role; its
        runtime.validation_opponents must list exactly the "validation"
        members in the expected order. Every case must share validation_maps,
        red_zone_depth and confirmation_seed_pairs.
    opponent_bundle : str or Path
        Local bundle folder with pool.json and the actor folders it names. Its
        members must match the expected opponents one for one, in order.
    output_dir : str or Path
        Chosen workflow folder. protocol/ is written inside it; train the study
        into output_dir/study so the grouping's run folders match. It must stay
        outside the bundle.
    expected_opponents : str or Path or None, default=None
        Committed list of expected members (see _expected_opponents). None
        uses opponents.yaml beside the declaration.
    recipe_selection : str or Path or None, default=None
        Optional JSON or YAML file whose only key, recipe_selection, is the
        list copied into grouping.yaml. Each pick names declared case IDs as
        groups; training.analyze checks its rules. None writes no picks.

    Returns
    -------
    Path
        Absolute path of output_dir/protocol/study.yaml.

    Raises
    ------
    ValueError
        Paths overlap the bundle or use symbolic links; an input file is
        missing or malformed; a bundle member's identity differs; the
        declaration's opponents, panel or shared settings break the rules
        above; preview of the bound declaration fails; pool.json changed during
        preparation; or a different protocol/ already exists.
    TypeError
        The bound declaration has a setting of the wrong type.
    OSError
        A file cannot be read.

    Notes
    -----
    Host-only. Each actor's files are hashed and the validation panel is built
    once in a temporary folder to read its labels and digest; no game,
    learning update or worker starts. Set JAX_PLATFORMS=cpu for this command.
    Rerunning with the same inputs reuses an identical protocol/. Move the
    bundle with the workflow folder to keep the relative paths valid.

    See Also
    --------
    prepare_software_check : The small six-method wiring check.
    marl_battlegrounds.training.preview_study : Lists the prepared runs.
    """
    from marl_battlegrounds.experiments.hydra import (
        _preparation_paths,  # pyright: ignore[reportPrivateUsage]
    )

    bundle, root = _preparation_paths(opponent_bundle, output_dir)
    from marl_battlegrounds import training
    from marl_battlegrounds.training._launch import read_declaration
    from marl_battlegrounds.training.checkpoints import runtime_identity
    from marl_battlegrounds.training.validation import VALIDATION_MAPS

    source_file = Path(declaration).expanduser().resolve()
    identities_file = (
        source_file.parent / "opponents.yaml"
        if expected_opponents is None
        else Path(expected_opponents).expanduser().resolve()
    )
    picks_file = (
        None
        if recipe_selection is None
        else Path(recipe_selection).expanduser().resolve()
    )
    inputs = [source_file, identities_file, *([picks_file] if picks_file else [])]
    for path in inputs:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Required declaration file is missing: {path}")
    raw_inputs = {path: path.read_bytes() for path in inputs}
    expected = _expected_opponents(identities_file)
    bundle, source, raw, pool = _read_opponent_pool(bundle)
    if type(pool.get("schema_version")) is not int or pool["schema_version"] != 1:
        raise ValueError("Opponent bundle has a different schema")
    references = _verify_opponent_members(
        bundle,
        pool,
        [
            {key: value for key, value in member.items() if key != "roles"}
            for member in expected
        ],
        count_message=(
            f"Opponent bundle must hold exactly the {len(expected)} expected members"
        ),
    )
    destination = root / "protocol"
    relative = {
        member["name"]: os.path.relpath(references[member["name"]], destination)
        if member["kind"] == "actor"
        else member["name"]
        for member in expected
    }
    trainers = {m["name"] for m in expected if "training" in m["roles"]}
    panel_names = [m["name"] for m in expected if "validation" in m["roles"]]
    study = read_declaration(source_file)
    cases = study.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Study cases must be a nonempty list")
    for raw_case in cast(list[Any], cases):
        if not isinstance(raw_case, dict) or "config" not in raw_case:
            raise ValueError("Every declared study case must be a training case")
        case = cast(dict[str, Any], raw_case)
        config = case["config"]
        runtime = case.get("runtime")
        if not isinstance(config, dict) or not isinstance(runtime, dict):
            raise ValueError(f"Case {case.get('id')} needs config and runtime objects")
        config, runtime = cast(dict[str, Any], config), cast(dict[str, Any], runtime)
        if set(runtime) != {"validation_opponents"} or any(
            config.get(name) is not None
            for name in ("validation_opponents", "validation_panel")
        ):
            raise ValueError(
                f"Case {case.get('id')} must name its panel only in "
                "runtime.validation_opponents; preparation binds opponents"
            )
        opponents = config.get("opponents")
        if (
            not isinstance(opponents, dict)
            or set(cast(dict[str, Any], opponents)) - {"self", "past"} != trainers
        ):
            raise ValueError(
                f"Case {case.get('id')} opponents must be self, past and exactly "
                "the expected training opponents"
            )
        if runtime["validation_opponents"] != panel_names:
            raise ValueError(
                f"Case {case.get('id')} validation opponents must be the expected "
                f"panel in order: {panel_names}"
            )
        case["runtime"] = {
            "opponents": {
                name: relative[name]
                for name in cast(dict[str, Any], opponents)
                if name in relative and relative[name] != name
            },
            "validation_opponents": [relative[name] for name in panel_names],
        }
    cases = cast(list[dict[str, Any]], cases)
    # The study owner checks every setting, ID, seed and resource now. Preview
    # only parses paths; the bound paths were checked above.
    training.preview_study(study, output_dir=root / "study")
    maps = _same_setting(cases, "validation_maps", list(VALIDATION_MAPS))
    depth = _same_setting(cases, "red_zone_depth", 5.0)
    pairs = _same_setting(cases, "confirmation_seed_pairs", 50)
    # The shared panel owner determines labels and identity. Its path is not part
    # of the digest; each run independently creates the same declared panel.
    with tempfile.TemporaryDirectory(prefix="marl-declared-panel-") as temporary:
        panel = training.create_panel(
            opponents=[references[name] for name in panel_names],
            output_dir=temporary,
        )
    run_rows: list[dict[str, str]] = []
    design = io.StringIO()
    writer = csv.DictWriter(design, fieldnames=("case_id", "recipe", "method", "seed"))
    writer.writeheader()
    for case in cases:
        for seed in case["seeds"]:
            full_id = f"{case['id']}--seed-{seed}"
            run_rows.append(
                {
                    "run_dir": f"../study/cases/{full_id}/run",
                    "group": case["id"],
                    "unit": f"seed-{seed}",
                }
            )
            writer.writerow(
                {
                    "case_id": full_id,
                    "recipe": case["id"],
                    "method": case["config"]["method"],
                    "seed": seed,
                }
            )
    grouping: dict[str, Any] = {
        "schema_version": 2,
        "measure": "point_margin",
        "maps": maps,
        "opponents": [member.name for member in panel.members],
        "panel_digest": panel.digest,
        "red_zone_depth": float(depth),
        "runs": run_rows,
        "comparisons": [],
    }
    if picks_file is not None:
        picks = read_declaration(picks_file)
        if set(picks) != {"recipe_selection"} or not isinstance(
            picks["recipe_selection"], list
        ):
            raise ValueError("Recipe picks need one recipe_selection list")
        grouping["recipe_selection"] = picks["recipe_selection"]
    selection = {
        "schema_version": 1,
        "name": f"{study['name']} Point Margin",
        "rule": "point_margin",
        "confirmation_seed_pairs": pairs,
    }

    hashed = {**raw_inputs, source: raw}

    def record(path: Path) -> dict[str, str]:
        """Name one input relative to protocol/ with its exact byte hash."""
        return {
            "path": os.path.relpath(path, destination),
            "sha256": hashlib.sha256(hashed[path]).hexdigest(),
        }

    methods = sorted({case["config"]["method"] for case in cases})
    records = {
        "study.yaml": OmegaConf.to_yaml(OmegaConf.create(study)),
        "grouping.yaml": OmegaConf.to_yaml(OmegaConf.create(grouping)),
        "checkpoint-selection.yaml": OmegaConf.to_yaml(OmegaConf.create(selection)),
        "design.csv": design.getvalue(),
        "source.json": json.dumps(
            {
                "schema_version": 1,
                "meaning": (
                    "Committed study declaration bound to a checked opponent "
                    "bundle; no training or games have run"
                ),
                "declaration": record(source_file),
                "expected_opponents": record(identities_file),
                "opponent_pool": record(source),
                **(
                    {"recipe_selection": record(picks_file)}
                    if picks_file is not None
                    else {}
                ),
                "members": expected,
                "runtime_identities": {
                    method: runtime_identity(method) for method in methods
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    }
    if source.read_bytes() != raw or any(
        path.read_bytes() != value for path, value in raw_inputs.items()
    ):
        raise ValueError("A preparation input changed during preparation")
    return _publish_protocol(root, records)


@hydra.main(version_base="1.3", config_path="configs", config_name="prepare")
def main(config: DictConfig) -> Path:
    """Prepare one protocol folder and return/print its exact declaration path.

    Parameters
    ----------
    config : DictConfig
        Packaged prepare settings. protocol chooses the route:
        "analysis_software_check" copies synthetic saved rows into output_dir;
        "software_check" also needs opponent_bundle; "declared_study" needs
        declaration and opponent_bundle, and reads the optional
        expected_opponents (default: opponents.yaml beside the declaration)
        and recipe_selection files. Relative command paths use the original
        working directory; "~" expands to the home folder.

    Returns
    -------
    Path
        The grouping YAML for analysis_software_check, otherwise
        output_dir/protocol/study.yaml. The same path is printed.

    Raises
    ------
    ValueError
        The protocol is unknown, a required input is missing, or the chosen
        route refuses its inputs. Failures happen before protocol output is
        written.

    Notes
    -----
    No route launches a study, trains or plays games.
    """

    def path(value: object) -> str:
        """Expand "~" and resolve one command path from the launch folder."""
        return to_absolute_path(str(Path(str(value)).expanduser()))

    output = path(config.output_dir)
    if config.protocol == "analysis_software_check":
        result = prepare_analysis_software_check(output)
    elif config.protocol == "software_check":
        if not config.opponent_bundle:
            raise ValueError(
                "software_check needs opponent_bundle=/path/frozen-opponents"
            )
        result = prepare_software_check(path(config.opponent_bundle), output)
    elif config.protocol == "declared_study":
        if not config.get("declaration") or not config.opponent_bundle:
            raise ValueError(
                "declared_study needs declaration=/path/study.yaml and "
                "opponent_bundle=/path/opponents"
            )
        expected = config.get("expected_opponents")
        picks = config.get("recipe_selection")
        result = prepare_declared_study(
            path(config.declaration),
            path(config.opponent_bundle),
            output,
            expected_opponents=path(expected) if expected else None,
            recipe_selection=path(picks) if picks else None,
        )
    else:
        raise ValueError(
            "Choose protocol=declared_study, software_check or analysis_software_check"
        )
    print(result)
    return result


if __name__ == "__main__":
    prepare_hydra()
    main()
