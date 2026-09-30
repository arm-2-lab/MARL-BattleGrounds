"""Generate packaged Hydra settings from the existing configuration records.

Run ``python scripts/dev/generate_experiment_configs.py`` from the checkout, or
add ``--check`` to reject drift without writing. Requires the hydra extra.
Only packaged YAML changes. This tool starts no numerical backend or training;
software_check values exercise wiring and are not paper hyperparameters.
"""

from __future__ import annotations

import argparse
import ast
import json
from copy import deepcopy
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from marl_battlegrounds.baselines.config import PPOConfig, PQNConfig, QMIXConfig
from marl_battlegrounds.baselines.methods import TRAINING_METHODS, method_settings_field
from marl_battlegrounds.training.config import TrainConfig

ROOT = (
    Path(__file__).resolve().parents[2] / "src/marl_battlegrounds/experiments/configs"
)


def _defaults(record: type[Any]) -> dict[str, Any]:
    """Copy declared dataclass defaults into JSON values, retaining sentinels.

    record is one of the four host setting dataclasses. Resolve field factories,
    not the enclosing constructor: method-dependent None values must reach the
    API validator. Reject non-JSON defaults instead of inventing YAML encodings.
    """
    values: dict[str, Any] = {}
    for field in fields(record):
        value: Any = field.default
        if field.default_factory is not MISSING:
            value = field.default_factory()
        values[field.name] = (
            asdict(value)
            if is_dataclass(value) and not isinstance(value, type)
            else value
        )
    return json.loads(json.dumps(values, allow_nan=False))


def _example_presets() -> dict[str, dict[str, Any]]:
    """Read literal tutorial defaults without importing a numerical example.

    Each named lesson owns its keyword arguments. Required arguments become
    OmegaConf missing values; tuples become YAML lists. Refuse a nonliteral
    default rather than duplicate it here. This reads source files only.
    """
    examples = (
        "mappo_training",
        "qmix_training",
        "pqn_training",
        "curriculum",
        "competitive_training",
        "partner_training",
        "own_learner",
        "training_collection",
        "training_distributions",
        "training_continuation",
        "training_selection",
        "training_study",
        "evaluation",
        "evaluation_results",
        "scenario_evaluation",
        "cross_play_and_zsc",
        "population_selection",
        "llm",
        "canonical_tournament",
    )
    result: dict[str, dict[str, Any]] = {}
    for name in examples:
        entry = "launch" if name == "cross_play_and_zsc" else "run"
        tree = ast.parse((ROOT.parents[3] / "examples" / f"{name}.py").read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == entry
        )
        arguments = function.args
        pairs = list(
            zip(
                arguments.args,
                [None] * (len(arguments.args) - len(arguments.defaults))
                + arguments.defaults,
                strict=True,
            )
        )
        pairs.extend(zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True))
        defaults = {
            argument.arg: "???" if default is None else ast.literal_eval(default)
            for argument, default in pairs
        }
        result[f"examples/{name}.yaml"] = {
            "defaults": ["_self_", {"/resources": "inherit"}],
            **json.loads(json.dumps(defaults, allow_nan=False)),
            "preview": False,
            "hydra": {
                "callbacks": {
                    "launch": {
                        "_target_": (
                            "marl_battlegrounds.experiments.hydra._prepare_launch"
                        ),
                        "mode": "${hydra.mode}",
                        "output_dir": "${oc.select:output_dir,null}",
                        "resume_from": None,
                        "launch_dir": "${hydra.run.dir}",
                        "records_subdir": "${hydra.output_subdir}",
                    }
                },
                "job": {"chdir": False},
                "run": {
                    "dir": "${marl_launch_dir:${oc.select:output_dir,null},"
                    + name
                    + ",null}"
                },
                "output_subdir": ".hydra",
            },
        }
        if name == "scenario_evaluation":
            hydra = result[f"examples/{name}.yaml"]["hydra"]
            callback = hydra["callbacks"]["launch"]
            callback["resume_from"] = "${resume_from}"
            callback["scenario_inputs"] = {
                "scenario_ids": "${scenario_ids}",
                "system": "${system}",
                "opponent": "${opponent}",
                "resources": "${resources}",
            }
            hydra["run"]["dir"] = (
                "${marl_launch_dir:${output_dir},scenario_evaluation,null,${resume_from}}"
            )
        if name == "canonical_tournament":
            hydra = result[f"examples/{name}.yaml"]["hydra"]
            hydra["callbacks"]["launch"]["additional_output_dir"] = "${directory}"
            hydra["run"]["dir"] = (
                "${marl_launch_dir:${directory},canonical_tournament,null,${output_dir}}"
            )
    return result


def presets() -> dict[str, dict[str, Any]]:
    """Return every packaged preset as plain data derived from shared defaults.

    Keys are relative YAML filenames. Values contain one complete root or group.
    The software recipe changes only declared software-check values. No actor
    paths, remote downloads, paper ranges or hardware assumptions are inserted.
    """
    base = _defaults(TrainConfig)
    for field in ("method", "ppo", "qmix", "pqn"):
        base.pop(field)
    result: dict[str, dict[str, Any]] = {
        "train.yaml": {
            "defaults": [
                "_self_",
                {"method": "mappo"},
                {"recipe": "default"},
                {"recipe_settings": "${recipe}_${method}"},
                {"pool": "none"},
                {"curriculum_preset": "none"},
                {"resources": "inherit"},
                {"override hydra/launcher": "study"},
            ],
            **base,
            "output_dir": None,
            "preview": False,
            "runtime": {},
            "name": "training",
            "study_options": {},
            "hydra": {
                "callbacks": {
                    "launch": {
                        "_target_": (
                            "marl_battlegrounds.experiments.hydra._prepare_launch"
                        ),
                        "mode": "${hydra.mode}",
                        "output_dir": "${output_dir}",
                        "resume_from": "${oc.select:resume_from,null}",
                        "require_resume": False,
                        "launch_dir": "${hydra.run.dir}",
                        "records_subdir": "${hydra.output_subdir}",
                        "sweep_dir": "${hydra.sweep.dir}",
                        "launcher_target": "${hydra.launcher._target_}",
                    }
                },
                "job": {"chdir": False},
                "run": {
                    "dir": (
                        "${marl_launch_dir:${output_dir},${name},"
                        "${oc.select:resume_from,null}}"
                    )
                },
                "sweep": {"dir": "${hydra.run.dir}", "subdir": "unused"},
                "output_subdir": ".hydra",
            },
        },
        "hydra/launcher/study.yaml": {
            "_target_": "hydra_plugins.marl_battlegrounds_study.StudyLauncher"
        },
        "recipe/default.yaml": {"recipe": "default"},
        "recipe/software_check.yaml": {"recipe": "software_check"},
        "pool/none.yaml": {"opponent_pool": None},
        "pool/scripted.yaml": {
            "opponent_pool": {
                "self": 0.6,
                "past": 0.2,
                "tdm-alpha": 1 / 15,
                "tdm-beta": 1 / 15,
                "tdm-gamma": 1 / 15,
            }
        },
        "curriculum_preset/none.yaml": {},
        "curriculum_preset/staged.yaml": {"curriculum": True},
        "resources/inherit.yaml": {"resources": None},
        "resources/cpu.yaml": {
            "resources": {"devices": [{"device": "cpu", "max_workers": 1}]}
        },
    }
    evaluation_defaults: dict[str, Any] = {
        "num_episodes": 32,
        "spawn_mode": "paired",
        "num_envs": 32,
        "keep_batch_size": True,
        "metrics": "priority",
        "replay_episodes": [],
        "save_replays": 0,
        "output_dir": None,
        "phase": "evaluation",
        "pass_id": "1",
        "chunk_size": 16,
    }
    evaluation_hydra = {
        "callbacks": {
            "launch": {
                "_target_": "marl_battlegrounds.experiments.hydra._prepare_launch",
                "mode": "${hydra.mode}",
                "output_dir": "${oc.select:evaluation.output_dir,null}",
                "resume_from": "${oc.select:evaluation.resume_from,null}",
                "launch_dir": "${hydra.run.dir}",
                "records_subdir": "${hydra.output_subdir}",
                "evaluation_inputs": {
                    "checkpoint": "${checkpoint}",
                    "opponent": "${opponent}",
                    "resources": "${resources}",
                },
            }
        },
        "job": {"chdir": False},
        "run": {
            "dir": (
                "${marl_launch_dir:${oc.select:evaluation.output_dir,null},"
                "${name},null}"
            )
        },
        "output_subdir": ".hydra",
    }
    result["evaluate.yaml"] = {
        "defaults": ["_self_", {"evaluation": "default"}, {"resources": "inherit"}],
        "checkpoint": "???",
        "opponent": "tdm-alpha",
        "preview": False,
        "name": "evaluation",
        "hydra": evaluation_hydra,
    }
    result["evaluation/default.yaml"] = {"evaluation": evaluation_defaults}
    result["evaluation/full.yaml"] = {
        "evaluation": {
            **evaluation_defaults,
            "metrics": "full",
            "replay_episodes": [1, 2],
        }
    }
    result["evaluate_resume.yaml"] = {
        "checkpoint": "???",
        "opponent": "???",
        "preview": False,
        "resources": None,
        "name": "evaluation",
        "evaluation": {
            "resume_from": "???",
            "num_envs": 32,
            "keep_batch_size": True,
            "phase": "evaluation",
            "pass_id": "1",
            "chunk_size": 16,
        },
        "hydra": deepcopy(evaluation_hydra),
    }
    stage_hydra = {
        "job": {"chdir": False},
        "run": {
            "dir": (
                "${marl_launch_dir:${output_dir},stage,null,"
                "${oc.select:resume_from,null}}"
            )
        },
        "output_subdir": ".hydra",
        "callbacks": {
            "launch": {
                "_target_": "marl_battlegrounds.experiments.hydra._prepare_launch",
                "mode": "${hydra.mode}",
                "output_dir": "${output_dir}",
                "resume_from": "${oc.select:resume_from,null}",
                "launch_dir": "${hydra.run.dir}",
                "records_subdir": "${hydra.output_subdir}",
            }
        },
    }
    result["prepare.yaml"] = {
        "protocol": "analysis_software_check",
        "opponent_bundle": None,
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    result["prepare.yaml"]["hydra"]["callbacks"]["launch"]["opponent_bundle"] = (
        "${opponent_bundle}"
    )
    result["study.yaml"] = {
        "declaration": None,
        "output_dir": None,
        "resume_from": None,
        "preview": False,
        "hydra": deepcopy(stage_hydra),
    }
    result["select.yaml"] = {
        "run_dirs": [],
        "declaration": None,
        "output_dir": None,
        "resume_from": None,
        "confirm": False,
        "hydra": deepcopy(stage_hydra),
    }
    result["analyze.yaml"] = {
        "study_dirs": [],
        "output_dir": "???",
        "grouping": None,
        "selection": None,
        "evaluation_dirs": [],
        "analysis_data_dir": None,
        "protocol": None,
        "hydra": deepcopy(stage_hydra),
    }
    result["resume.yaml"] = {
        "output_dir": None,
        "resume_from": "???",
        "preview": False,
        "runtime": {},
        "resources": None,
        "name": "training",
        "study_options": {},
        "hydra": deepcopy(result["train.yaml"]["hydra"]),
    }
    result["resume.yaml"]["hydra"]["callbacks"]["launch"]["require_resume"] = True
    settings = {"ppo": PPOConfig, "qmix": QMIXConfig, "pqn": PQNConfig}
    for method in TRAINING_METHODS:
        field = method_settings_field(method)
        result[f"method/{method}.yaml"] = {
            "method": method,
            field: _defaults(settings[field]),
        }
        result[f"recipe_settings/default_{method}.yaml"] = {}
        software: dict[str, Any] = {
            "seed": 710,
            "num_envs": 32,
            "total_env_steps": 20480,
            "curriculum": False,
            "keep_past": 2,
            "history_capture_env_steps": [9600, 19200],
            "validation_fractions": [1.0],
            "routine_seed_pairs": 1,
            "confirmation_seed_pairs": 1,
            "checkpoint_env_steps": [20480],
            "checkpoint_interval_updates": 1000000,
            "retain_all_actors": True,
            "recording": False,
            "metrics": "priority",
        }
        if field == "ppo":
            software[field] = {
                "rollout_length": 4,
                "epochs": 1,
                "groups": 2,
                "minibatches": 2,
            }
        elif field == "qmix":
            software[field] = {
                "rollout_length": 4,
                "buffer_size": 16,
                "min_buffer_size": 4,
                "sample_sequence_length": 4,
                "sample_batch_size": 4,
                "epochs": 1,
            }
        else:
            software[field] = {
                "rollout_length": 4,
                "memory_window": 2,
                "epochs": 1,
                "num_minibatches": 2,
            }
        result[f"recipe_settings/software_check_{method}.yaml"] = software
    result.update(_example_presets())
    # Tutorials have a separate search root so their plain method setting does
    # not collide with training's method group. Both use these shared presets.
    for name in ("inherit", "cpu"):
        result[f"examples/resources/{name}.yaml"] = result[f"resources/{name}.yaml"]
    return result


def main() -> None:
    """Write generated YAML, or fail --check when content or inventory differs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = presets()
    stale: list[str] = []
    for name, value in expected.items():
        destination = ROOT / name
        header = "# Generated by scripts/dev/generate_experiment_configs.py.\n"
        if "/" in name and not name.startswith("hydra/"):
            header = "# @package _global_\n" + header
        text = header + OmegaConf.to_yaml(OmegaConf.create(value), sort_keys=False)
        if args.check:
            if not destination.is_file() or destination.read_text() != text:
                stale.append(name)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text)
    extra = {
        p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.yaml")
    } - expected.keys()
    if stale or extra:
        raise SystemExit(f"Packaged configuration drift: {sorted(set(stale) | extra)}")


if __name__ == "__main__":
    main()
