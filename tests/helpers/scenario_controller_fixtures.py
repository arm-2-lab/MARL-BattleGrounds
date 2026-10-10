"""Load fixed historical controller inputs without the private authoring tools.

The JSON inputs retain the exact old config, state and source digests. Core builds
matching observations and masks. Original draft JSON remains unchanged for source
metadata checks and private authoring tests. No current scenario replaces an old
fixture, and no expected actions or trajectories are generated here.
"""

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.env import initialize_scenario_state
from marl_battlegrounds.core.types import (
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    ResolvedAgentProfile,
)
from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v2
from marl_battlegrounds.tasks import (
    _initial_state_digest,  # pyright: ignore[reportPrivateUsage]
)


@dataclass(frozen=True, slots=True)
class ControllerScenario:
    config: EnvConfig
    initial_state: EnvState
    observation: Observation
    action_mask: ActionMask
    info: Info
    semantic_digest: str | None = None
    map_semantic_digest: str | None = None
    resolved_configuration_digest: str | None = None
    resolved_initial_state_digest: str | None = None


def _array(value: dict[str, Any]) -> Array:
    return jnp.asarray(value["values"], dtype=value["dtype"]).reshape(value["shape"])


@cache
def load_controller_input(name: str) -> ControllerScenario:
    path = Path(__file__).parents[1] / "fixtures" / "controller_inputs" / f"{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload["source"]
    if "path" in source:
        original = Path(__file__).parents[2] / source["path"]
        if hashlib.sha256(original.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Historical source JSON does not match its saved input")
    config_values: dict[str, Any] = {
        name: _array(cast(dict[str, Any], value)) if isinstance(value, dict) else value
        for name, value in payload["config"].items()
        if name != "agent_profile"
    }
    config_values["agent_profile"] = ResolvedAgentProfile(
        **{
            name: _array(value)
            for name, value in payload["config"]["agent_profile"].items()
        }
    )
    config = EnvConfig(**config_values)
    authored = EnvState(
        **{name: _array(value) for name, value in payload["initial_state"].items()}
    )
    state, observation, mask, info = initialize_scenario_state(authored, config)
    if (
        build_resolved_env_config_v2(config).canonical_digest_sha256
        != payload["resolved_configuration_digest"]
        or _initial_state_digest(state) != payload["resolved_initial_state_digest"]
    ):
        raise ValueError("Historical controller input does not match its saved digests")
    return ControllerScenario(
        config,
        state,
        observation,
        mask,
        info,
        **{
            name: payload[name]
            for name in (
                "semantic_digest",
                "map_semantic_digest",
                "resolved_configuration_digest",
                "resolved_initial_state_digest",
            )
        },
    )


SCENARIO_1_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "scenario_1_r34.json"
SCENARIO_1_SEMANTIC_DIGEST = (
    "d30c9fe5ac5fbe82e8831bdb2726cfbd3a2bf89a0749ac97ed492f79eb47f031"
)
SCENARIO_1_MAP_DIGEST = (
    "ac87824928b74f7db555b73a0127b56942f7747a6ee704c498924bf317149c63"
)
SCENARIO_1_STATE_DIGEST = (
    "45039793469def83242f64dc01a68c68a7a3f030f917e2baae69a7557d29c20c"
)


def load_scenario_1() -> ControllerScenario:
    return load_controller_input("scenario_1")


SCENARIO_3_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "scenario_3_r9.json"
SCENARIO_3_SEMANTIC_DIGEST = (
    "03fafb3201c7cf9c18674a71d97b943b9e92162b9d445aed80e3b7724249f063"
)


def load_scenario_3() -> ControllerScenario:
    return load_controller_input("scenario_3")


SCENARIO_5_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "scenario_5_r9.json"
SCENARIO_5_SEMANTIC_DIGEST = (
    "20361030c886f508778305e79fd161b93595db407972a92bcd0a6d3d90760fd9"
)


def load_scenario_5() -> ControllerScenario:
    return load_controller_input("scenario_5")


# Scenario 6 r10 supplied the geometry and state. Its author-approved correction
# changes Team B's starting score from 18 to 19; this is not byte-identical r10.
SCENARIO_6_FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "scenario_6_r10_score_17_19.json"
)
SCENARIO_6_SEMANTIC_DIGEST = (
    "6ad5b3ffa18de9defd77d8da7d3601cec7d8fceb1ea0e156f0b857c86d8331bb"
)


def load_scenario_6() -> ControllerScenario:
    return load_controller_input("scenario_6")


SCENARIO_7_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "scenario_7_r25.json"
SCENARIO_7_SEMANTIC_DIGEST = (
    "455117f81ff2b7dc8bb9f3d6abee294810c21e2af58fde61f237b20ec33e042e"
)


def load_scenario_7() -> ControllerScenario:
    return load_controller_input("scenario_7")


SCENARIO_8_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "scenario_8_r14.json"
SCENARIO_8_SEMANTIC_DIGEST = (
    "cb25125624f0f61e6e160c9a8e408d5301554c0c018a734339405b3a7c3e7926"
)


def load_scenario_8() -> ControllerScenario:
    return load_controller_input("scenario_8")
