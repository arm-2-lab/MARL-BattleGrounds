"""Generate packaged Hydra settings from the existing configuration records.

Run ``python scripts/dev/generate_experiment_configs.py`` from the checkout, or add
``--check`` to reject drift without writing. Requires the hydra extra. Only packaged
YAML changes. This tool starts no numerical backend or training; software_check values
exercise wiring and are not paper hyperparameters. The default recipe holds the settings
of each method's tournament policy run (SELECTED_RECIPES;
scripts/paper_experiments/hyperparameter_search.md, Final Policies): its seed, learner
settings, games, reward and 250,675,200-transition budget, and the search's training and
validation settings. The dataclass defaults are unchanged.
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
_PPO_SELECTED = {
    "actor_lr": 0.0001,
    "critic_lr": 0.00025,
    "rollout_length": 32,
    "epochs": 4,
    "minibatches": 2,
    "groups": 2,
    "gamma": 0.999,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coefficient": 0.01,
    "value_coefficient": 0.5,
    "max_grad_norm": 0.5,
    "adam_epsilon": 1e-05,
    "input_scale": 0.01,
    "value_normalization": True,
}
_PER_AGENT_REWARD = "marl_battlegrounds.training.reward_recipes:combat_support_reward"
_TEAM_REWARD = "marl_battlegrounds.training.reward_recipes:combat_support_team_reward"
# Training and validation settings shared by every tournament policy run. Each
# ran 250,675,200 transitions: 40,108,032 with routine checks every 10%, then
# a continuation with 20 checks, one every 5% of the added budget. As shares of
# the whole budget those are 0.016 to 0.16 in steps of 0.016, then 0.202 to 1.0
# in steps of 0.042 (30 checks). The six fixed recurrent MAPPO training models
# and the pooled 250M validation model are not packaged, so the opponents and
# the validation panel keep the API defaults.
_TOURNAMENT_PROTOCOL: dict[str, Any] = {
    "total_env_steps": 250675200,
    "keep_past": 12,
    "past_capture_interval": 1376256,
    "validation_maps": [42, 43, 44, 46],
    "validation_fractions": [
        *(round(0.016 * i, 3) for i in range(1, 11)),
        *(round(0.16 + 0.042 * i, 3) for i in range(1, 21)),
    ],
    "routine_seed_pairs": 4,
    "confirmation_seed_pairs": 8,
    "retain_all_actors": True,
}
# The settings of each method's tournament policy run, with its seed
# (scripts/paper_experiments/hyperparameter_search.md, Final Policies). recipe=default
# applies them over the dataclass defaults; every value is that run's saved
# one. For QMIX that run's recipe differs from the method's final recipe
# (target copies every 128 updates instead of reward scale 10).
SELECTED_RECIPES: dict[str, dict[str, Any]] = {
    "mappo": {
        **_TOURNAMENT_PROTOCOL,
        "seed": 20261002,
        "num_envs": 512,
        "reward": _PER_AGENT_REWARD,
        "ppo": _PPO_SELECTED,
    },
    "ippo": {
        **_TOURNAMENT_PROTOCOL,
        "seed": 20261001,
        "num_envs": 512,
        "reward": _PER_AGENT_REWARD,
        "ppo": _PPO_SELECTED,
    },
    "ff_mappo": {
        **_TOURNAMENT_PROTOCOL,
        "seed": 12259871,
        "num_envs": 512,
        "reward": _TEAM_REWARD,
        "ppo": {**_PPO_SELECTED, "actor_lr": 0.0003},
    },
    "ff_ippo": {
        **_TOURNAMENT_PROTOCOL,
        "seed": 20261001,
        "num_envs": 512,
        "reward": _PER_AGENT_REWARD,
        "ppo": _PPO_SELECTED,
    },
    "qmix": {
        **_TOURNAMENT_PROTOCOL,
        "seed": 12259871,
        "num_envs": 768,
        "reward": _TEAM_REWARD,
        "qmix": {
            "rollout_length": 32,
            "buffer_size": 172,
            "min_buffer_size": 32,
            "sample_sequence_length": 20,
            "sample_batch_size": 128,
            "epochs": 48,
            "q_lr": 0.0001,
            "hard_update": True,
            "update_period": 128,
            "tau": 0.01,
            "gamma": 0.99,
            "eps_min": 0.05,
            "eps_decay": 2000000,
            "input_scale": 0.01,
            "reward_scale": 1.0,
        },
    },
    "pqn_vdn": {
        **_TOURNAMENT_PROTOCOL,
        "seed": 12259871,
        "num_envs": 768,
        "reward": _TEAM_REWARD,
        "pqn": {
            "rollout_length": 128,
            "memory_window": 4,
            "epochs": 4,
            "num_minibatches": 12,
            "q_lr": 0.00025,
            "lr_linear_decay": True,
            "max_grad_norm": 1.0,
            "gamma": 0.99,
            "td_lambda": 0.85,
            "eps_start": 1.0,
            "eps_finish": 0.01,
            # The tournament run explored over 5% of its 40M run's 407 planned
            # learning blocks (20.35 blocks, about 2M transitions); this is the
            # share of a 250M run's 2,549 blocks that gives the same span.
            "eps_decay_fraction": 0.00798352295017654,
            "input_scale": 1.0,
            "reward_scale": 10.0,
            "normalization": "batch_norm",
        },
    },
}


# The Season 0 field in rank order: the six tournament actors (relative to models_dir,
# the hyperparameter-search root; scripts/paper_experiments/hyperparameter_search.md,
# "The six tournament policies") and the three scripted systems, then Random as a
# reference floor that is not a member of the field.
SEASON_0_FIELD = (
    (
        "Recurrent IPPO",
        "${models_dir}/250m/study/cases/ippo-reference--seed-20261001/run/actors/"
        "ippo_actor_step_000250675200_76b55aa1d72003c22020d13872730d9cf6d96b50a6a936026287b7a35f0f3328",
    ),
    (
        "PQN-VDN",
        "${models_dir}/250m/study/cases/pqn_vdn-b768-t128-reward-x10--seed-12259871/run/actors/"
        "pqn_vdn_actor_step_000240156672_8c1e52fbccfb238801b31041eb4df87cf6df38c62cc458e379c00d6224036f30",
    ),
    (
        "Recurrent MAPPO",
        "${models_dir}/250m/study/cases/mappo-reference--seed-20261002/run/actors/"
        "mappo_actor_step_000219103232_01f7e47402a9ba3e6a17a302ad9f21e680647cbb75e0186c7f1ef12f67303174",
    ),
    (
        "QMIX",
        "${models_dir}/250m/study/cases/qmix-b768-t32-lr-1e-4-target-faster--seed-12259871/run/actors/"
        "qmix_actor_step_000219095040_c9a2ada94944055f57ebfafdc87245954bd270c603a48bbe1a634bf2761cf499",
    ),
    (
        "Feedforward IPPO",
        "${models_dir}/250m/study/cases/ff_ippo-reference--seed-20261001/run/actors/"
        "ff_ippo_actor_step_000219103232_83cc8f75bd015ea02ec977a4e446afbe3a4f247fddb604c42b60f8edcebc1952",
    ),
    (
        "Feedforward MAPPO",
        "${models_dir}/ff-mappo-250m/study/cases/ff_mappo-team-reward--seed-12259871/run/actors/"
        "ff_mappo_actor_step_000208568320_f2b2650e9f35e96296d8b70bb28043dd551cc11abd382c75b366e9e24528a6f4",
    ),
    ("TDM-Beta", "tdm-beta"),
    ("TDM-Gamma", "tdm-gamma"),
    ("TDM-Alpha", "tdm-alpha"),
    ("Random", "random"),
)


# Season 0 display names for the nine entrants, keyed as the saved tournament
# records them. The teamwork, zero-shot and human-games analyses read them.
SEASON_0_RECORDED_NAMES = {
    "IPPO at step 250,675,200 (run)": "Recurrent IPPO",
    "PQN-VDN at step 240,156,672 (run)": "PQN-VDN",
    "MAPPO at step 219,103,232 (run)": "Recurrent MAPPO",
    "QMIX at step 219,095,040 (run)": "QMIX",
    "FF-IPPO at step 219,103,232 (run)": "Feedforward IPPO",
    "FF-MAPPO at step 208,568,320 (run)": "Feedforward MAPPO",
    "tdm-beta": "TDM-Beta",
    "tdm-gamma": "TDM-Gamma",
    "tdm-alpha": "TDM-Alpha",
}


# The training history of each Season 0 baseline in rank order: its 40M run, then the
# run that continued it to 250M, relative to runs_dir, the hyperparameter-search root
# (scripts/paper_experiments/hyperparameter_search.md, "The six tournament policies").
SEASON_0_LINEAGES = (
    ("Recurrent IPPO", "shapes-and-ppo", "250m", "ippo-reference--seed-20261001"),
    (
        "PQN-VDN",
        "q-settings",
        "250m",
        "pqn_vdn-b768-t128-reward-x10--seed-12259871",
    ),
    ("Recurrent MAPPO", "shapes-and-ppo", "250m", "mappo-reference--seed-20261002"),
    (
        "QMIX",
        "q-settings",
        "250m",
        "qmix-b768-t32-lr-1e-4-target-faster--seed-12259871",
    ),
    ("Feedforward IPPO", "shapes-and-ppo", "250m", "ff_ippo-reference--seed-20261001"),
    (
        "Feedforward MAPPO",
        "ff-mappo-levers",
        "ff-mappo-250m",
        "ff_mappo-team-reward--seed-12259871",
    ),
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
        "declaration": None,
        "expected_opponents": None,
        "recipe_selection": None,
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
    result["scenarios.yaml"] = {
        "defaults": ["_self_"],
        "systems": [
            {"name": "TDM-Beta", "method": "tdm-beta"},
            {"name": "TDM-Gamma", "method": "tdm-gamma"},
            {"name": "TDM-Alpha", "method": "tdm-alpha"},
            {"name": "Random", "method": "random"},
        ],
        "models_dir": None,
        "scenario_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "num_episodes": 256,
        "num_envs": 128,
        "seed": 20261006,
        "metrics": "priority",
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    result["scenario_analysis.yaml"] = {
        "scenarios_dir": "???",
        "output_dir": "???",
        "reference": "Random",
        "hydra": deepcopy(stage_hydra),
    }
    result["llm_analysis.yaml"] = {
        "field_dir": None,
        "scenarios_dir": None,
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    # The paper supplement's example prompt: TDM-Alpha against TDM-Beta on
    # test map 47, from the first tick at or after 12 with a possible hit.
    result["llm_prompt.yaml"] = {
        "map_id": 47,
        "team_a": "tdm-alpha",
        "team_b": "tdm-beta",
        "seed": 0,
        "min_tick": 12,
        "max_ticks": 120,
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    # The qualified local Qwen recipe (docs/llm.md) with the 9B
    # model; the server is started separately with the recipe's settings.
    result["llm_challenge.yaml"] = {
        "defaults": ["_self_"],
        "systems": [
            {"name": "TDM-Beta", "method": "tdm-beta"},
            {"name": "TDM-Gamma", "method": "tdm-gamma"},
            {"name": "TDM-Alpha", "method": "tdm-alpha"},
        ],
        "models_dir": None,
        "skip": ["Random"],
        "model": "qwen3.5-9b-fp8",
        "server_url": "http://127.0.0.1:8000/v1",
        "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "history_turns": 0,
        "failure_policy": "fallback",
        "records": "light",
        "maps": [47, 48, 49, 50, 51],
        "games_per_map": 2,
        "seed": 20261007,
        "max_steps": 300,
        "metrics": "full",
        "time_limit_hours": 8.0,
        # The scenarios stage's Season 0 settings: 256 games per scenario.
        "scenario_ids": [],
        "scenario_games": 256,
        "scenario_envs": 128,
        "scenario_seed": 20261006,
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    learned = [name for name, _ in SEASON_0_FIELD[:6]]
    result["zero_shot_coordination.yaml"] = {
        "defaults": [{"field": "season_0"}, "_self_"],
        "host": learned[0],
        "newcomers": [*learned[1:], "Random"],
        "classes": ["Mage", "Warrior", "Hunter", "Rogue", "Priest"],
        "opponents": [name for name, _ in SEASON_0_FIELD[:9]],
        "maps": [47, 48, 49, 50, 51],
        "num_episodes": 200,
        "seed": 20261007,
        "num_envs": 128,
        "metrics": "full",
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    # Home games: the Season 0 tournament replayed with full metrics by the
    # tournament_analysis stage, read by display name like the teamwork study.
    result["zero_shot_coordination_analysis.yaml"] = {
        "zsc_dir": "???",
        "bootstrap_draws": 2000,
        "bootstrap_seed": 20261007,
        "home_games_dir": None,
        "home_names": dict(SEASON_0_RECORDED_NAMES),
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    tournament_winner, runner_up = SEASON_0_FIELD[0][0], SEASON_0_FIELD[1][0]
    result["human_games.yaml"] = {
        "defaults": [{"field": "season_0"}, "_self_"],
        "parts": [
            {
                "kind": "against",
                "opponent": tournament_winner,
                "maps": [47, 48, 49],
                "seed": 20261007,
            },
            {
                "kind": "alongside",
                "opponent": runner_up,
                "partner": tournament_winner,
                "player_class": "mage",
                "maps": [47, 48, 49],
                "seed": 20261013,
            },
        ],
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    # The series names Systems by display name, so the analysis maps the
    # field's recorded names to them.
    result["human_games_analysis.yaml"] = {
        "series_dir": "???",
        "field_dir": "???",
        "names": dict(SEASON_0_RECORDED_NAMES),
        "window_ticks": 10,
        "lead_margin": 12,
        # Luck for Ultimates used together: random slides from a declared seed.
        "chance_shuffles": 2000,
        "chance_seed": 20261016,
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    result["field/season_0.yaml"] = {
        "models_dir": "???",
        "systems": [
            {"name": name, "method": method} for name, method in SEASON_0_FIELD
        ],
    }
    result["speed.yaml"] = {
        "map_id": 47,
        "batch_sizes": [32, 64, 128, 512, 1024],
        "steps": 500,
        "repeats": 5,
        "seed": 0,
        "train_method": "mappo",
        "train_num_envs": 1024,
        "train_rollout_length": 32,
        "train_blocks": 40,
        "warmup_blocks": 3,
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    result["training_curves.yaml"] = {
        "lineages": "???",
        "opponents": None,
        "output_dir": "???",
        "hydra": deepcopy(stage_hydra),
    }
    result["training_runs/season_0.yaml"] = {
        "runs_dir": "???",
        "lineages": [
            {
                "name": name,
                "runs": [
                    f"${{runs_dir}}/{study}/study/cases/{case}/run"
                    for study in (first, continued)
                ],
            }
            for name, first, continued, case in SEASON_0_LINEAGES
        ],
    }
    # Season 0: the nine entrants' display names and the scripted bots, which
    # the figure draws as open points and the screen's learned correlation
    # leaves out.
    result["tournament_analysis.yaml"] = {
        "tournament_dir": "???",
        "field_file": "???",
        "names": dict(SEASON_0_RECORDED_NAMES),
        "scripted": ["TDM-Beta", "TDM-Gamma", "TDM-Alpha"],
        "num_envs": 128,
        "replay_dir": None,
        "output_dir": "???",
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
        result[f"recipe_settings/default_{method}.yaml"] = deepcopy(
            SELECTED_RECIPES[method]
        )
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
