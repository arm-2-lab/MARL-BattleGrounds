"""Give live debugger episodes explicit evaluation identities and seed records.

The launcher builds one ``DebuggerEvaluationLaunchSpecificationV1``. Session
creation and deliberate restarts call ``build_debugger_evaluation_context_v1`` with
the actual effective config and an increasing run generation. The result records
nonofficial live provenance, both team controllers and the information mode,
and, when the session passes its slot labels, every controller's name (per team
and per champion slot) so a saved replay shows the same names as live play.
Fixed packaged scenarios retain their exact setup and evaluator seed protocol;
ordinary diagnostics retain their recorded scalar environment seed.
These builders validate and hash host data; they do not discover Git state, run a
transition, or write artifacts.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, StringConstraints, model_validator

from marl_battlegrounds._battle_client.model import (
    DebuggerScenario,
    ScenarioMode,
    TeamBController,
    TeamController,
    TeamControllerActionSource,
    is_system_controller,
    is_team_controller,
    team_controller_action_source,
)
from marl_battlegrounds.core.config import validate_env_config
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    TASK_MODE_TDM,
    TEAM_A_ID,
    EnvConfig,
)
from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V4,
    SHARED_OBS_ACTOR_PROJECTION_V3,
)
from marl_battlegrounds.evaluation.catalog import (
    build_evaluation_episode_context_v4,
    build_evaluation_seed_protocol_v1,
    build_resolved_env_config_v2,
)
from marl_battlegrounds.evaluation.controller_names import (
    slot_controller_name_keys,
    team_controller_name_key,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV2,
    CodeRevisionV1,
    CodeRevisionV2,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV4,
    EvaluationEpisodeIdentityV1,
    EvaluationModel,
    EvaluationRole,
    ExecutionInformationMode,
    NotApplicablePolicySlotV1,
    PolicyAssignmentSlotV2,
    ResolvedEnvConfigV2,
    VersionedIdentityV1,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.recording_context import recorded_policy_assignment
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    reactive_tdm_alpha_controller_descriptor,
)
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
)
from marl_battlegrounds.policies.reactive_tdm_gamma import (
    reactive_tdm_gamma_controller_descriptor,
)

DEBUGGER_EVALUATION_BRIDGE_SCHEMA_VERSION: Literal[1] = 1
DEBUGGER_EVALUATION_LAUNCH_SPECIFICATION_SCHEMA_ID = (
    "marl_battlegrounds.visual_debugger.evaluation_launch_specification"
)
DEBUGGER_PUBLIC_AGENT_IDS_V1 = tuple(str(slot) for slot in range(MAX_AGENT_SLOTS))

type DebuggerActionSourceKindV1 = TeamControllerActionSource
type DebuggerCaptureProfileV1 = Literal["debug", "evaluation_metric_complete"]

_Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]
_NonNegativeInt = Annotated[int, Field(ge=0)]


class DebuggerEvaluationLaunchSpecificationV1(EvaluationModel):
    """Explicit launch-scoped inputs shared by every restarted episode."""

    schema_id: Literal[
        "marl_battlegrounds.visual_debugger.evaluation_launch_specification"
    ] = DEBUGGER_EVALUATION_LAUNCH_SPECIFICATION_SCHEMA_ID
    schema_version: Literal[1] = DEBUGGER_EVALUATION_BRIDGE_SCHEMA_VERSION
    specification_id: Annotated[
        str,
        StringConstraints(pattern=r"^debugger-evaluation-launch:[0-9a-f]{64}$"),
    ]
    launch_content_digest_sha256: _Sha256Hex
    canonical_digest_sha256: _Sha256Hex
    root_seed: _Seed
    code_revision: CodeRevisionV1 | CodeRevisionV2
    capture_profile: DebuggerCaptureProfileV1

    @model_validator(mode="after")
    def _validate_launch_specification(
        self,
    ) -> DebuggerEvaluationLaunchSpecificationV1:
        """Require both content digests and the launch ID to match the supplied
        launch fields.
        """
        launch_payload = {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "root_seed": self.root_seed,
            "code_revision": self.code_revision,
            "capture_profile": self.capture_profile,
        }
        expected_content_digest = canonical_digest_sha256(launch_payload)
        if self.launch_content_digest_sha256 != expected_content_digest:
            raise ValueError("debugger launch content digest mismatch")
        if self.specification_id != (
            f"debugger-evaluation-launch:{expected_content_digest}"
        ):
            raise ValueError("debugger launch specification ID is not canonical")
        if self.canonical_digest_sha256 != canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        ):
            raise ValueError("debugger launch specification digest mismatch")
        return self


def build_debugger_evaluation_launch_specification_v1(
    *,
    root_seed: int,
    code_revision: CodeRevisionV1 | CodeRevisionV2,
    capture_profile: DebuggerCaptureProfileV1 = "debug",
) -> DebuggerEvaluationLaunchSpecificationV1:
    """Build the immutable launch record shared by restarted debugger episodes.

    Parameters
    ----------
    root_seed : int
        Root seed in the unsigned 32-bit range, from zero through 2**32 - 1.
    code_revision : CodeRevisionV1 or CodeRevisionV2
        Already discovered source identity. This function does not inspect Git itself.
    capture_profile : {"debug", "evaluation_metric_complete"}, optional
        Recorded capture contract, default ``debug``.

    Returns
    -------
    DebuggerEvaluationLaunchSpecificationV1
        Strict launch record with matching content digest, specification ID and digest.

    Raises
    ------
    ValueError
        If a seed, capture choice or identity fails strict model validation.
    """
    launch_payload = {
        "schema_id": DEBUGGER_EVALUATION_LAUNCH_SPECIFICATION_SCHEMA_ID,
        "schema_version": DEBUGGER_EVALUATION_BRIDGE_SCHEMA_VERSION,
        "root_seed": root_seed,
        "code_revision": code_revision,
        "capture_profile": capture_profile,
    }
    launch_content_digest = canonical_digest_sha256(launch_payload)
    payload = {
        **launch_payload,
        "specification_id": f"debugger-evaluation-launch:{launch_content_digest}",
        "launch_content_digest_sha256": launch_content_digest,
    }
    payload["canonical_digest_sha256"] = canonical_digest_sha256(payload)
    return DebuggerEvaluationLaunchSpecificationV1.model_validate(payload)


def _scenario_contract_payload(scenario: DebuggerScenario) -> dict[str, object]:
    """Project stable authored metadata without serializing its state callback.

    A packaged map with exchanged spawn banks adds
    ``packaged_map_spawn_banks_exchanged: True``; every other scenario keeps
    its earlier payload, so its scenario identity is unchanged.
    """
    if type(scenario) is not DebuggerScenario:
        raise TypeError("scenario must be the exact DebuggerScenario type")
    payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.visual_debugger.scenario_contract",
        "schema_version": 1,
        "name": scenario.name,
        "title": scenario.title,
        "description": scenario.description,
        "mode": scenario.mode,
        "audience": scenario.audience,
        "default_controlled_slot": scenario.default_controlled_slot,
        "frames": tuple(
            {
                "label": frame.label,
                "description": frame.description,
                "commands": tuple(
                    {
                        "actor_global_slot": command.actor_global_slot,
                        "move_action": command.move_action,
                        "target_global_slot": command.target_global_slot,
                        "use_ultimate": command.use_ultimate,
                    }
                    for command in frame.commands
                ),
            }
            for frame in scenario.frames
        ),
    }
    provenance = scenario.provenance
    if provenance is not None:
        payload["authored_provenance"] = {
            "source_kind": provenance.source_kind,
            "source_identity": provenance.source_identity,
            "scenario_semantic_digest": provenance.scenario_semantic_digest,
            "map_semantic_digest": provenance.map_semantic_digest,
            "resolved_configuration_digest": provenance.resolved_configuration_digest,
            "resolved_initial_state_digest": provenance.resolved_initial_state_digest,
        }
    if scenario.scenario_specification is not None:
        payload["packaged_scenario"] = scenario.scenario_specification
    if scenario.packaged_map is not None:
        payload["packaged_map"] = scenario.packaged_map
    if scenario.spawn_banks_exchanged:
        payload["packaged_map_spawn_banks_exchanged"] = True
    return payload


def _packaged_map_keys(
    scenario: DebuggerScenario,
    config: EnvConfig,
    resolved_config: ResolvedEnvConfigV2,
) -> tuple[AggregationKeyV1, ...] | None:
    """Check a packaged map's geometry and return its recording keys.

    Parameters
    ----------
    scenario : DebuggerScenario
        A scenario whose packaged_map is set (the caller checks this).
    config : EnvConfig
        The exact scalar runtime config of the episode.
    resolved_config : ResolvedEnvConfigV2
        ``build_resolved_env_config_v2(config)``, already built by the caller.

    Returns
    -------
    tuple of AggregationKeyV1 or None
        Without exchanged banks: the four registered map keys (map_id,
        map_name, map_origin "registered", map_split). With
        ``scenario.spawn_banks_exchanged``: the same four keys, checked against
        the config with its spawn banks put back in catalog order, then
        ``marl_bgs.source_config_id`` (the content ID of that put-back config)
        and ``marl_bgs.spawn_locations`` "1", the keys evaluate records for
        exchanged spawns (``recording_context._map_recording_keys``). None when
        the exchanged map fails that check; the caller then records a custom
        map and never claims the packaged map.

    Raises
    ------
    ValueError
        Without exchanged banks, the config's geometry (size, obstacles or
        ordered spawn banks) differs from the packaged map. This is the earlier
        behavior for every packaged map.

    Notes
    -----
    Host-only. It reads the packaged catalog and hashes the config; it starts
    no episode and writes no file.
    """
    from marl_battlegrounds.evaluation.map_identity import registered_map_metadata

    packaged_map = scenario.packaged_map
    if packaged_map is None:
        raise ValueError("a packaged map is required")
    if not scenario.spawn_banks_exchanged:
        # Frozen Pydantic models provide their runtime hash for this host cache.
        return registered_map_metadata(packaged_map.map_id, resolved_config)  # pyright: ignore[reportArgumentType]
    from marl_battlegrounds.evaluation.evaluation_conditions import config_record
    from marl_battlegrounds.tasks import (
        _swap_spawn_banks,  # pyright: ignore[reportPrivateUsage]
    )

    source = _swap_spawn_banks(config)
    try:
        keys = registered_map_metadata(
            packaged_map.map_id,
            build_resolved_env_config_v2(source),  # pyright: ignore[reportArgumentType]
        )
    except ValueError:
        return None
    return (
        *keys,
        AggregationKeyV1(
            name="marl_bgs.source_config_id", value=config_record(source)[0]
        ),
        AggregationKeyV1(name="marl_bgs.spawn_locations", value="1"),
    )


def _content_identity(
    identifier: str,
    payload: dict[str, object],
) -> ContentAddressedIdentityV1:
    """Build a version-one content identity from a named payload and its SHA-256
    digest.
    """
    return ContentAddressedIdentityV1(
        identifier=identifier,
        version=1,
        canonical_digest=canonical_digest_sha256(payload),
    )


def _derive_named_seed(
    root_seed: int,
    *,
    namespace: str,
) -> int:
    """Derive one launch-stable seed for exact same-start comparisons."""
    payload = canonical_json_bytes(
        {
            "schema_id": "marl_battlegrounds.visual_debugger.named_seed",
            "schema_version": 1,
            "root_seed": root_seed,
            "namespace": namespace,
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _action_source_contract_payload(
    *,
    action_source_kind: DebuggerActionSourceKindV1,
    scenario_mode: ScenarioMode,
    team_a_controller: TeamController,
    team_b_controller: TeamBController,
    execution_information_mode: ExecutionInformationMode,
    scenario_contract_digest: str,
    reactive_tdm_identity: ContentAddressedIdentityV1 | None = None,
    scenario_controller_identity: ContentAddressedIdentityV1 | None = None,
    team_a_scenario_controller_identity: ContentAddressedIdentityV1 | None = None,
) -> dict[str, object]:
    """Describe both team action sources and their input contract for stable
    provenance.

    Scripted scenarios use schema version 1. Interactive sessions use version
    4; version 5 adds ``scenario_5_execution_included`` when Team B is BETA
    (``scenario_5``), and version 6 adds ``tdm_gamma_execution_included`` when
    Team B is GAMMA (``tdm_gamma``). Versions 5 and 6 need
    ``scenario_controller_identity``: the content address of Team B's
    controller descriptor and code revision. When Team A is BETA or GAMMA the
    payload is version 7: it adds ``team_a_scenario_controller`` (Team A's
    content address, required) and sets ``scenario_5_execution_included`` and
    ``tdm_gamma_execution_included`` from both teams. Combinations without BETA
    or GAMMA on Team A keep their exact version 4, 5 or 6 bytes. Versions 2 and
    3 are retired and not reused. The payload is only hashed; nothing reads its
    version back.
    """
    if action_source_kind not in ("manual", "scripted", "mixed", "policy"):
        raise ValueError(
            "action_source_kind must be manual, scripted, mixed, or policy"
        )
    if scenario_mode == "scripted":
        # Fixed-frame diagnostic identities must remain byte-for-byte V1.
        return {
            "schema_id": "marl_battlegrounds.visual_debugger.action_source_contract",
            "schema_version": 1,
            "action_source_kind": action_source_kind,
            "team_b_controller": team_b_controller,
            "execution_information_mode": execution_information_mode,
            "manual_submission_included": action_source_kind in ("manual", "mixed"),
            "scripted_submission_included": action_source_kind in ("scripted", "mixed"),
            "scenario_contract_digest_sha256": scenario_contract_digest,
            "policy_execution_included": False,
        }
    controllers = (team_a_controller, team_b_controller)
    if "reactive_tdm" in controllers and reactive_tdm_identity is None:
        raise ValueError("Reactive TDM action source requires its controller identity")
    if (
        team_b_controller in ("scenario_5", "tdm_gamma")
        and scenario_controller_identity is None
    ):
        raise ValueError("Scenario action source requires its controller identity")
    payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.visual_debugger.action_source_contract",
        "schema_version": 4,
        "action_source_kind": action_source_kind,
        "team_a_controller": team_a_controller,
        "team_b_controller": team_b_controller,
        "execution_information_mode": execution_information_mode,
        "manual_submission_included": "manual" in controllers,
        "reactive_tdm_execution_included": "reactive_tdm" in controllers,
        "random_policy_execution_included": "random_valid" in controllers,
        # Retain the existing V4/V5 field without retaining an executable controller.
        "scenario_3_execution_included": False,
        "reactive_tdm_controller": reactive_tdm_identity,
        "scenario_controller": scenario_controller_identity,
        "scenario_contract_digest_sha256": scenario_contract_digest,
        "policy_execution_included": any(
            controller != "manual" for controller in controllers
        ),
    }
    if team_b_controller == "scenario_5":
        payload["schema_version"] = 5
        payload["scenario_5_execution_included"] = True
    elif team_b_controller == "tdm_gamma":
        # GAMMA-only V6: V4's fields plus its own execution flag. The GAMMA
        # descriptor and code revision are content-addressed in
        # scenario_controller. V1, V4 and V5 bytes are unchanged.
        payload["schema_version"] = 6
        payload["tdm_gamma_execution_included"] = True
    if team_a_controller in ("scenario_5", "tdm_gamma"):
        # V7: BETA or GAMMA on Team A. Earlier combinations never reach here.
        if team_a_scenario_controller_identity is None:
            raise ValueError("Team A scenario action source requires its identity")
        payload["schema_version"] = 7
        payload["team_a_scenario_controller"] = team_a_scenario_controller_identity
        payload["scenario_5_execution_included"] = "scenario_5" in controllers
        payload["tdm_gamma_execution_included"] = "tdm_gamma" in controllers
    return payload


def _policy_assignments(
    config: EnvConfig,
    scenario: DebuggerScenario,
    *,
    action_source_kind: DebuggerActionSourceKindV1,
    team_a_controller: TeamController,
    team_b_controller: TeamBController,
    actor_projection: VersionedIdentityV1,
    action_contract_digest: str,
    reactive_tdm_identity: ContentAddressedIdentityV1 | None = None,
    scenario_controller_identity: ContentAddressedIdentityV1 | None = None,
    team_a_scenario_controller_identity: ContentAddressedIdentityV1 | None = None,
    system_registrations: tuple[
        tuple[str, dict[str, object]], tuple[str, dict[str, object]]
    ]
    | None = None,
    manual_slots: tuple[int, ...] | None = None,
) -> tuple[PolicyAssignmentSlotV2, ...]:
    """Assign each active fixed slot its recorded role and action-source identity.

    Inactive slots remain unassigned. Controller identities and actor projection
    describe execution; they do not introduce a learner or policy registry entry.
    Explicit Manual rows override the team System assignment. Configured dead
    rows keep their assignment; only unconfigured rows are not applicable.
    BETA and GAMMA rows use their own team's identity:
    ``team_a_scenario_controller_identity`` for Team A and
    ``scenario_controller_identity`` for Team B.
    """
    profile = config.agent_profile
    active = np.asarray(profile.active_mask, dtype=np.bool_)
    team_ids = np.asarray(profile.team_ids, dtype=np.int32)
    focal_slot = scenario.default_controlled_slot
    if not bool(active[focal_slot]):
        raise ValueError("scenario default actor must be configured active")
    focal_team_id = int(team_ids[focal_slot])
    rows: list[PolicyAssignmentSlotV2] = []
    for slot in range(MAX_AGENT_SLOTS):
        if not bool(active[slot]):
            rows.append(NotApplicablePolicySlotV1(global_slot=slot))
            continue
        if scenario.scenario_specification is not None:
            fixed_role = scenario.scenario_specification.role_template[slot]
            if fixed_role == "not_applicable":
                raise ValueError(
                    "Every active fixed scenario slot needs its declared role"
                )
            role: EvaluationRole = fixed_role
        elif slot == focal_slot:
            role = "focal"
        elif int(team_ids[slot]) == focal_team_id:
            role = "cooperative_partner"
        else:
            role = "adversarial_opponent"
        team_id = int(team_ids[slot])
        if scenario.mode == "scripted":
            policy_kind = "scripted"
        else:
            policy_kind = (
                team_a_controller if team_id == TEAM_A_ID else team_b_controller
            )
        if manual_slots is not None and slot in manual_slots:
            policy_kind = "manual"
        if is_system_controller(policy_kind):
            if system_registrations is None:
                raise ValueError("System assignment needs its registration")
            identifier, registration = system_registrations[
                0 if team_id == TEAM_A_ID else 1
            ]
            rows.append(
                recorded_policy_assignment(
                    slot=slot,
                    role=role,
                    phase="evaluation",
                    name=registration["name"],
                    descriptor=registration,
                    system_id=identifier,
                    registration=registration,
                )
            )
            continue
        controller_identity = None
        team_scenario_identity = (
            team_a_scenario_controller_identity
            if team_id == TEAM_A_ID
            else scenario_controller_identity
        )
        if policy_kind == "reactive_tdm":
            algorithm_id = "reactive-team-deathmatch-controller"
            execution_mode = "deterministic"
            controller_identity = (
                team_scenario_identity
                if scenario.scenario_specification is not None and team_id != TEAM_A_ID
                else reactive_tdm_identity
            )
        elif policy_kind == "random_valid":
            algorithm_id = "canonical-random-valid"
            execution_mode = "stochastic"
        elif policy_kind == "scenario_5":
            algorithm_id = "scenario-5-pressure-controller"
            execution_mode = "deterministic"
            controller_identity = team_scenario_identity
        elif policy_kind == "tdm_gamma":
            algorithm_id = "reactive-team-deathmatch-gamma-controller"
            execution_mode = "deterministic"
            controller_identity = team_scenario_identity
        else:
            algorithm_id = "not_applicable"
            execution_mode = "deterministic"
        rows.append(
            AssignedPolicySlotV2(
                global_slot=slot,
                lifecycle="frozen",
                evaluation_role=role,
                policy_kind=policy_kind,
                policy_id=f"debugger-action-source:{policy_kind}:slot:{slot}",
                policy_content_digest=(
                    controller_identity.canonical_digest
                    if controller_identity is not None
                    else action_contract_digest
                ),
                checkpoint_digest=None,
                algorithm_id=algorithm_id,
                training_run_id="not_applicable",
                training_step=0,
                population_member_id=None,
                parameter_sharing_group_id=(
                    f"debugger-action-source:{policy_kind}:team:{int(team_ids[slot])}"
                ),
                preprocessing=actor_projection,
                normalization=VersionedIdentityV1(
                    identifier="none",
                    version=1,
                ),
                # Manual input is not a sampled policy; deterministic records
                # the lack of policy RNG rather than repeatable human choices.
                # Random alone samples from its seeded action distribution.
                execution_mode=execution_mode,
            )
        )
    return tuple(rows)


def debugger_action_source_kind_v1(
    scenario: DebuggerScenario,
    team_a_controller: TeamController,
    team_b_controller: TeamBController,
    *,
    manual_slots: tuple[int, ...] | None = None,
) -> DebuggerActionSourceKindV1:
    """Describe the actual action source for a scenario and its team controllers.

    Parameters
    ----------
    scenario : DebuggerScenario
        Scenario metadata, including interactive or scripted mode.
    team_a_controller : TeamController
        Selected Team A controller.
    team_b_controller : TeamBController
        Selected Team B controller; both teams accept the same kinds.

    manual_slots : tuple of int or None, optional
        Configured physical Manual rows, including dead agents. None uses the
        older whole-team choices. A Manual row inside a System makes it mixed.

    Returns
    -------
    DebuggerActionSourceKindV1
        Scripted for registered playback; otherwise manual, policy or mixed according
        to which teams use manual control. Individual controller IDs remain recorded.
    """
    if scenario.mode == "scripted":
        return "scripted"
    return team_controller_action_source(
        team_a_controller, team_b_controller, manual_slots=manual_slots
    )


def build_debugger_evaluation_context_v1(
    launch_specification: DebuggerEvaluationLaunchSpecificationV1,
    *,
    scenario: DebuggerScenario,
    config: EnvConfig,
    run_generation: int,
    action_source_kind: DebuggerActionSourceKindV1,
    team_a_controller: TeamController,
    team_b_controller: TeamBController,
    execution_information_mode: ExecutionInformationMode,
    expected_horizon: int | None = None,
    system_registrations: tuple[
        tuple[str, dict[str, object]], tuple[str, dict[str, object]]
    ]
    | None = None,
    manual_slots: tuple[int, ...] | None = None,
    slot_labels: tuple[str | None, ...] | None = None,
) -> EvaluationEpisodeContextV4:
    """Build the recorded custom-evaluation context for one live debugger episode.

    Parameters
    ----------
    launch_specification : DebuggerEvaluationLaunchSpecificationV1
        Exact validated launch record reused across deliberate episode replacements.
    scenario : DebuggerScenario
        Current scenario metadata and any exact authored-source provenance.
    config : EnvConfig
        Exact effective runtime config after caller choices have been applied.
    run_generation : int
        Nonnegative Python integer increased for each deliberate episode replacement.
    action_source_kind : DebuggerActionSourceKindV1
        Must agree with the scenario mode and both selected controllers.
    team_a_controller : TeamController
        Supported Team A controller selection.
    team_b_controller : TeamBController
        Supported Team B controller selection.
    execution_information_mode : {"shared_obs", "no_shared_obs"}
        Input contract for action selection. Reactive controllers require an
        interactive SharedObs session.
    expected_horizon : int or None, optional
        Positive captured-transition bound no greater than config.max_steps. None uses
        config.max_steps; callers pass script length or remaining interactive steps.

    system_registrations : tuple of registration pairs or None
        Optional Team A/B (ID, metadata) pairs already frozen and registered at
        setup. Required for declared System aliases; forbidden for built-in-only,
        scripted or NoSharedObs sessions. The bridge never loads a method.
        System routes use action payload V8 and one recorded System seed root;
        public System helpers split it by episode, operation and team. Control
        folds the accepted frame index into each action root. Old routes retain
        their original payloads and role-specific seeds.

    manual_slots : tuple of int or None, optional
        Ordered configured Manual rows. Mixed rows inside a System use action
        payload V9 and override their team System assignment. Existing whole-team
        rows retain their old identity bytes. None also preserves older whole-team
        identities. Invalid, duplicate, inactive or out-of-order rows are rejected.

    slot_labels : tuple of str or None, or None, optional
        The session's ten controller names in slot order, exactly as the live
        clients show them ("Manual" for a human-played slot, a method's own
        name otherwise; None for an unused slot). When given, the context
        records them so a saved replay keeps every controller's name: one
        ``marl_bgs.controller_name.slot_N`` key per used slot, and one
        ``marl_bgs.system_name.team_a``/``team_b`` key per team holding its
        slots' different names joined with " / " in slot order (the keys
        evaluate writes). None (the default) records no names, as before.
        The names never enter episode, matchup or seed identities.

    Returns
    -------
    EvaluationEpisodeContextV4
        Matching roster, config, task/scenario identity, controller assignments,
        named seed streams, capture choices and custom/nonofficial metadata.
        When the scenario provenance names an approved map ID, the layout
        identity is that map's approved source: its catalog asset ID, catalog
        revision and semantic digest, and the aggregation keys record the map as
        registered. A packaged map gets the same layout identity and keys. If
        the scenario declares the packaged map's spawn banks exchanged, the
        map is checked with the banks put back and the keys also record the
        exchange (``marl_bgs.source_config_id`` and
        ``marl_bgs.spawn_locations`` "1"); a failed check records a custom map
        with the resolved-debugger-environment layout instead. Other
        authored provenance gives the layout identity
        ``authored-map`` version 1 with the map's semantic digest, recorded as a
        custom map. Without provenance the layout identity is
        ``resolved-debugger-environment`` version 1 with the resolved config digest.

        Four aggregation keys name the BETA (``scenario_5``) and GAMMA
        (``tdm_gamma``) controllers; each pair appears only when its team uses
        one of them, and both pairs appear when both teams do:

        - ``pressure_protocol``: Team B's controller as ``policy_id@version``,
          for example ``scenario-5-pressure-controller@5``. Present only when
          Team B is BETA or GAMMA.
        - ``pressure_protocol_digest``: the content digest of that Team B
          identity (its descriptor plus the launch code revision). Same rule.
        - ``team_a_pressure_protocol``: Team A's controller as
          ``policy_id@version``. Present only when Team A is BETA or GAMMA.
        - ``team_a_pressure_protocol_digest``: the content digest of that
          Team A identity. Same rule.

        Each BETA or GAMMA policy row records its own team's digest as
        ``policy_content_digest``.

    Raises
    ------
    TypeError
        If the launch record or config is not the exact supported model type.
    ValueError
        If generation, horizon, controllers, input mode, source classification,
        configuration or recorded identity is invalid, if the provenance map ID
        names an approved map whose semantic digest differs from the scenario's
        map digest, or if slot_labels does not give exactly the used slots a
        nonempty name.

    Notes
    -----
    The control layer uses the recorded evaluator streams for fixed scenarios;
    ordinary diagnostics use ``seed_protocol.environment_seed``.
    Launch-stable named seeds preserve the same initial conditions across repeated
    comparisons; run_generation changes episode identity without hiding new randomness.
    This builds host metadata and may copy arrays for validation/hashing. It does not
    start an episode or write a replay. An approved map's source identity is read
    from the packaged TDM catalog on the host. Only the layout digest enters the
    evaluation, matchup, match and episode IDs, so those IDs do not depend on the
    layout label.
    """
    if type(launch_specification) is not DebuggerEvaluationLaunchSpecificationV1:
        raise TypeError(
            "launch_specification must be the exact V1 launch specification"
        )
    launch = DebuggerEvaluationLaunchSpecificationV1.model_validate_json(
        launch_specification.model_dump_json()
    )
    if type(run_generation) is not int or run_generation < 0:
        raise ValueError("run_generation must be a nonnegative exact integer")
    if type(config) is not EnvConfig:
        raise TypeError("config must be the exact EnvConfig type")
    if not is_team_controller(team_a_controller):
        raise ValueError(
            "team_a_controller must be manual, reactive_tdm, random_valid, "
            "scenario_5, or tdm_gamma"
        )
    if not is_team_controller(team_b_controller):
        raise ValueError(
            "team_b_controller must be manual, reactive_tdm, random_valid, "
            "scenario_5, or tdm_gamma"
        )
    if any(
        controller in ("reactive_tdm", "scenario_5", "tdm_gamma")
        for controller in (team_a_controller, team_b_controller)
    ) and (
        execution_information_mode != "shared_obs" or scenario.mode != "interactive"
    ):
        raise ValueError("Reactive controllers require interactive SharedObs.")
    if execution_information_mode not in ("shared_obs", "no_shared_obs"):
        raise ValueError(
            "execution_information_mode must be shared_obs or no_shared_obs"
        )
    expected_action_source_kind = debugger_action_source_kind_v1(
        scenario,
        team_a_controller,
        team_b_controller,
        manual_slots=manual_slots,
    )
    if action_source_kind != expected_action_source_kind:
        raise ValueError(
            "action_source_kind must match scenario mode and team controllers"
        )
    system_route = any(
        is_system_controller(value) for value in (team_a_controller, team_b_controller)
    )
    if system_route != (system_registrations is not None):
        raise ValueError("System aliases require their registered setup methods")
    if system_route and (
        scenario.mode != "interactive" or execution_information_mode != "shared_obs"
    ):
        raise ValueError("Systems require interactive SharedObs")
    validate_env_config(config)
    if manual_slots is not None:
        active_slots = tuple(
            int(slot) for slot in np.flatnonzero(config.agent_profile.active_mask)
        )
        if (
            any(
                type(slot) is not int or slot not in active_slots
                for slot in manual_slots
            )
            or tuple(sorted(set(manual_slots))) != manual_slots
        ):
            raise ValueError(
                "Manual slots must be unique configured physical rows in order"
            )
        if scenario.scenario_specification is not None and any(
            slot >= 5 for slot in manual_slots
        ):
            raise ValueError(
                "Scenario opponents are fixed. Remove team_b. "
                "Only Team A controllers can be chosen."
            )
        if scenario.mode != "interactive" and manual_slots:
            raise ValueError("Scripted playback cannot own Manual slots")
    resolved_config = build_resolved_env_config_v2(config)
    packaged_map_keys = (
        None
        if scenario.packaged_map is None
        else _packaged_map_keys(scenario, config, resolved_config)
    )
    horizon = config.max_steps if expected_horizon is None else expected_horizon
    if type(horizon) is not int or not 0 < horizon <= config.max_steps:
        raise ValueError("expected_horizon must be an exact positive config bound")

    specification = scenario.scenario_specification
    if specification is not None and (
        specification.resolved_config_digest_sha256
        != resolved_config.canonical_digest_sha256
        or specification.horizon != horizon
        or specification.seed_schedule.realized_seed_protocols[0].root_seed
        != launch.root_seed
    ):
        raise ValueError(
            "Fixed scenario config, horizon and seed must match its specification"
        )
    scenario_payload = _scenario_contract_payload(scenario)
    scenario_digest = canonical_digest_sha256(scenario_payload)
    scenario_identity_digest = (
        scenario.provenance.scenario_semantic_digest
        if scenario.provenance is not None
        else scenario_digest
    )
    layout_identity_digest = (
        scenario.provenance.map_semantic_digest
        if scenario.provenance is not None
        else resolved_config.canonical_digest_sha256
    )
    if specification is not None:
        scenario_identity_digest = specification.canonical_digest_sha256
        layout_identity_digest = specification.layout.canonical_digest
    elif scenario.packaged_map is not None and packaged_map_keys is not None:
        layout_identity_digest = scenario.packaged_map.source.semantic_digest
    approved_source = None
    if scenario.packaged_map is not None and packaged_map_keys is not None:
        from marl_battlegrounds.tasks import list_tdm_maps

        if list_tdm_maps()[scenario.packaged_map.map_id] != scenario.packaged_map:
            raise ValueError("Packaged map metadata must match the installed catalog")
        approved_source = scenario.packaged_map.source
    if scenario.provenance is not None and scenario.provenance.map_id is not None:
        from marl_battlegrounds.tasks import list_tdm_maps

        # The layout label names the approved source whose semantic digest equals
        # the loaded content: the catalog's asset ID and revision, never the
        # draft's own file revision, which is 0 for an unsaved buffer and moves
        # when the same content is saved again.
        approved_source = list_tdm_maps()[scenario.provenance.map_id].source
        if approved_source.semantic_digest != layout_identity_digest:
            raise ValueError(
                "provenance map_id names an approved map whose semantic digest "
                "differs from the scenario's map digest"
            )
    if specification is not None:
        layout_identifier = specification.layout.identifier
        layout_version = specification.layout.version
    elif approved_source is not None:
        layout_identifier = approved_source.asset_id
        layout_version = approved_source.revision
    elif scenario.provenance is not None:
        layout_identifier = "authored-map"
        layout_version = 1
    else:
        layout_identifier = "resolved-debugger-environment"
        layout_version = 1
    actor_projection = (
        SHARED_OBS_ACTOR_PROJECTION_V3
        if execution_information_mode == "shared_obs"
        else NO_SHARED_OBS_ACTOR_PROJECTION_V4
    )

    def controller_identity(
        descriptor: dict[str, object],
    ) -> ContentAddressedIdentityV1:
        """Bind a controller behavior descriptor to this launch's recorded code
        revision.
        """
        return ContentAddressedIdentityV1.model_validate(
            {
                "identifier": descriptor["policy_id"],
                "version": descriptor["version"],
                "canonical_digest": canonical_digest_sha256(
                    {"behavior": descriptor, "code_revision": launch.code_revision}
                ),
            }
        )

    reactive_tdm_identity = (
        controller_identity(reactive_tdm_alpha_controller_descriptor())
        if "reactive_tdm" in (team_a_controller, team_b_controller)
        else None
    )

    def pressure_identity_for(
        controller: TeamBController,
    ) -> ContentAddressedIdentityV1 | None:
        """Return BETA's or GAMMA's content address for that controller.

        ``scenario_5`` gives BETA's identity and ``tdm_gamma`` gives GAMMA's,
        each bound to this launch's code revision. Every other controller
        (manual, ALPHA ``reactive_tdm`` and Random ``random_valid``) returns
        None.
        """
        if controller == "scenario_5":
            return controller_identity(reactive_tdm_beta_controller_descriptor())
        if controller == "tdm_gamma":
            return controller_identity(reactive_tdm_gamma_controller_descriptor())
        return None

    scenario_controller_identity = (
        specification.pressure_protocol
        if specification is not None
        else pressure_identity_for(team_b_controller)
    )
    team_a_scenario_controller_identity = pressure_identity_for(team_a_controller)
    action_payload = _action_source_contract_payload(
        action_source_kind=action_source_kind,
        scenario_mode=scenario.mode,
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        execution_information_mode=execution_information_mode,
        scenario_contract_digest=scenario_digest,
        reactive_tdm_identity=reactive_tdm_identity,
        scenario_controller_identity=scenario_controller_identity,
        team_a_scenario_controller_identity=team_a_scenario_controller_identity,
    )
    if system_registrations is not None:
        action_payload = {
            **action_payload,
            "schema_version": 8,
            "system_registrations": {
                team: system_registrations[index][0]
                for index, team in enumerate(("team_a", "team_b"))
            },
            "system_key_protocol": (
                specification.seed_schedule.realized_seed_protocols[
                    0
                ].seed_protocol.identifier
                if specification is not None
                else "debugger-system-root-v1"
            ),
        }
    if specification is not None:
        action_payload = {
            **action_payload,
            "key_protocol": specification.seed_schedule.realized_seed_protocols[
                0
            ].seed_protocol,
            "numeric_episode_id": scenario.numeric_episode_id,
        }
    legacy_manual_slots = tuple(
        slot
        for slot in range(MAX_AGENT_SLOTS)
        if bool(config.agent_profile.active_mask[slot])
        and (team_a_controller if slot < 5 else team_b_controller) == "manual"
    )
    if (
        scenario.mode == "interactive"
        and manual_slots is not None
        and manual_slots != legacy_manual_slots
    ):
        action_payload = {
            **action_payload,
            "schema_version": 9,
            "manual_slots": manual_slots,
            "manual_submission_included": bool(manual_slots),
        }
    action_digest = canonical_digest_sha256(action_payload)
    config_digest = resolved_config.canonical_digest_sha256
    evaluation_payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.visual_debugger.evaluation_assignment",
        "schema_version": 1,
        "launch_content_digest_sha256": launch.launch_content_digest_sha256,
        "scenario_contract_digest_sha256": scenario_digest,
        "scenario_identity_digest_sha256": scenario_identity_digest,
        "layout_identity_digest_sha256": layout_identity_digest,
        "resolved_config_digest_sha256": config_digest,
        "action_source_contract_digest_sha256": action_digest,
        "execution_information_mode": execution_information_mode,
        "expected_horizon": horizon,
    }
    evaluation_digest = canonical_digest_sha256(evaluation_payload)
    generation_payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.visual_debugger.episode_assignment",
        "schema_version": 1,
        "evaluation_assignment_digest_sha256": evaluation_digest,
        "run_generation": run_generation,
    }
    assignment_digest = canonical_digest_sha256(generation_payload)
    matchup_digest = canonical_digest_sha256(
        {
            "scenario_contract_digest_sha256": scenario_digest,
            "scenario_identity_digest_sha256": scenario_identity_digest,
            "layout_identity_digest_sha256": layout_identity_digest,
            "resolved_config_digest_sha256": config_digest,
            "action_source_contract_digest_sha256": action_digest,
            "execution_information_mode": execution_information_mode,
        }
    )

    evaluation_suite = _content_identity(
        "visual-debugger-custom-suite",
        {
            "schema_id": "marl_battlegrounds.visual_debugger.custom_suite",
            "schema_version": 1,
            "official": False,
            "audience": "researcher",
        },
    )
    experiment_manifest = _content_identity(
        "visual-debugger-custom-manifest",
        evaluation_payload,
    )
    task = _content_identity(
        "team_deathmatch"
        if config.task_mode == TASK_MODE_TDM
        else "visual-debugger-analysis-task",
        {
            "schema_id": "marl_battlegrounds.visual_debugger.task",
            "schema_version": 1,
            "official": False,
            "task_kind": (
                "team_deathmatch"
                if config.task_mode == TASK_MODE_TDM
                else "interactive_visual_analysis"
            ),
        },
    )
    layout = ContentAddressedIdentityV1(
        identifier=layout_identifier,
        version=layout_version,
        canonical_digest=layout_identity_digest,
    )
    scenario_identity = ContentAddressedIdentityV1(
        identifier=(
            specification.scenario_id
            if specification is not None
            else "authored-team-deathmatch-scenario"
            if scenario.provenance is not None and config.task_mode == TASK_MODE_TDM
            else f"custom-debugger-scenario:{scenario.name}"
        ),
        version=1 if specification is None else specification.scenario_version,
        canonical_digest=scenario_identity_digest,
    )
    identity = EvaluationEpisodeIdentityV1(
        run_id=f"debugger-run:{launch.launch_content_digest_sha256}",
        evaluation_id=f"debugger-evaluation:{evaluation_digest}",
        matchup_id=f"debugger-matchup:{matchup_digest}",
        match_id=f"debugger-match:{assignment_digest}",
        episode_id=f"debugger-episode:{assignment_digest}",
        paired_comparison_key=None,
        evaluation_suite=evaluation_suite,
        experiment_manifest=experiment_manifest,
        task=task,
        layout=layout,
        curriculum=None,
        scenario=scenario_identity,
    )

    assignments = _policy_assignments(
        config,
        scenario,
        action_source_kind=action_source_kind,
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        actor_projection=actor_projection,
        action_contract_digest=action_digest,
        reactive_tdm_identity=reactive_tdm_identity,
        scenario_controller_identity=scenario_controller_identity,
        team_a_scenario_controller_identity=team_a_scenario_controller_identity,
        system_registrations=system_registrations,
        manual_slots=manual_slots,
    )
    active_roles = {
        row.evaluation_role
        for row in assignments
        if isinstance(row, AssignedPolicySlotV2)
    }

    def seed(namespace: str) -> int:
        """Derive a repeatable named seed from the unchanged launch root seed."""
        return _derive_named_seed(
            launch.root_seed,
            namespace=namespace,
        )

    if specification is not None:
        seed_protocol = specification.seed_schedule.realized_seed_protocols[0]
    else:
        seed_protocol = build_evaluation_seed_protocol_v1(
            seed_protocol=VersionedIdentityV1(
                identifier="debugger-system-root-v1"
                if system_route
                else "debugger-namespaced-sha256-u32",
                version=1,
            ),
            root_seed=launch.root_seed,
            episode_seed=seed("episode"),
            layout_seed=seed("layout"),
            environment_seed=seed("environment"),
            focal_policy_seed=seed(
                "system-action-source" if system_route else "focal-action-source"
            ),
            evaluation_seed=seed("evaluation"),
            cooperative_partner_seed=(
                seed(
                    "system-action-source"
                    if system_route
                    else "cooperative-action-source"
                )
                if "cooperative_partner" in active_roles
                else "not_applicable"
            ),
            adversarial_opponent_seed=(
                seed(
                    "system-action-source"
                    if system_route
                    else "adversarial-action-source"
                )
                if "adversarial_opponent" in active_roles
                else "not_applicable"
            ),
            scenario_seed=seed("scenario"),
        )
    aggregation_keys = [
        AggregationKeyV1(name="action_source", value=action_source_kind),
        AggregationKeyV1(name="information_regime", value=execution_information_mode),
        AggregationKeyV1(name="scenario", value=scenario.name),
        AggregationKeyV1(
            name="scenario_kind",
            value="custom" if specification is None else "packaged",
        ),
        AggregationKeyV1(name="team_b_controller", value=team_b_controller),
        AggregationKeyV1(name="tool", value="visual_debugger"),
    ]
    if system_registrations is not None:
        for team, controller, (identifier, registration) in zip(
            ("team_a", "team_b"),
            (team_a_controller, team_b_controller),
            system_registrations,
            strict=True,
        ):
            if is_system_controller(controller):
                aggregation_keys.extend(
                    [
                        AggregationKeyV1(
                            name=f"marl_bgs.system_id.{team}", value=identifier
                        ),
                        AggregationKeyV1(
                            name=f"marl_bgs.parameter_status.{team}",
                            value=str(registration["parameter_status"]),
                        ),
                    ]
                )
    if slot_labels is not None:
        active = np.asarray(config.agent_profile.active_mask, dtype=np.bool_)
        if len(slot_labels) != MAX_AGENT_SLOTS or any(
            (label is not None) != bool(active[slot])
            for slot, label in enumerate(slot_labels)
        ):
            raise ValueError("slot_labels must name exactly the configured slots")
        aggregation_keys.extend(slot_controller_name_keys(slot_labels))
        for team, start in (("team_a", 0), ("team_b", 5)):
            names = tuple(
                dict.fromkeys(
                    label
                    for label in slot_labels[start : start + 5]
                    if label is not None
                )
            )
            if names:
                aggregation_keys.append(
                    team_controller_name_key(team, " / ".join(names))
                )
    if packaged_map_keys is not None:
        aggregation_keys.extend(packaged_map_keys)
    elif scenario.provenance is not None and scenario.provenance.map_id is not None:
        from marl_battlegrounds.evaluation.map_identity import registered_map_metadata

        # Frozen Pydantic models provide their runtime hash for this host cache.
        aggregation_keys.extend(
            registered_map_metadata(scenario.provenance.map_id, resolved_config)  # pyright: ignore[reportArgumentType]
        )
    else:
        aggregation_keys.append(
            AggregationKeyV1(
                name="map_origin",
                value="custom",
            )
        )
    if scenario.mode == "interactive":
        aggregation_keys.append(
            AggregationKeyV1(name="team_a_controller", value=team_a_controller)
        )
    if reactive_tdm_identity is not None:
        aggregation_keys.extend(
            (
                AggregationKeyV1(
                    name="reactive_tdm_controller",
                    value=f"{reactive_tdm_identity.identifier}@{reactive_tdm_identity.version}",
                ),
                AggregationKeyV1(
                    name="reactive_tdm_controller_digest",
                    value=reactive_tdm_identity.canonical_digest,
                ),
            )
        )
    if scenario_controller_identity is not None:
        aggregation_keys.extend(
            (
                AggregationKeyV1(
                    name="pressure_protocol",
                    value=f"{scenario_controller_identity.identifier}@{scenario_controller_identity.version}",
                ),
                AggregationKeyV1(
                    name="pressure_protocol_digest",
                    value=scenario_controller_identity.canonical_digest,
                ),
            )
        )
    if team_a_scenario_controller_identity is not None:
        aggregation_keys.extend(
            (
                AggregationKeyV1(
                    name="team_a_pressure_protocol",
                    value=f"{team_a_scenario_controller_identity.identifier}@{team_a_scenario_controller_identity.version}",
                ),
                AggregationKeyV1(
                    name="team_a_pressure_protocol_digest",
                    value=team_a_scenario_controller_identity.canonical_digest,
                ),
            )
        )
    if scenario.provenance is not None:
        aggregation_keys.extend(
            (
                AggregationKeyV1(
                    name="scenario_source",
                    value=scenario.provenance.source_identity,
                ),
                AggregationKeyV1(
                    name="scenario_digest",
                    value=scenario.provenance.scenario_semantic_digest,
                ),
                AggregationKeyV1(
                    name="map_digest",
                    value=scenario.provenance.map_semantic_digest,
                ),
            )
        )
    if specification is not None:
        source = specification.authored_initial_condition
        aggregation_keys.extend(
            (
                AggregationKeyV1(
                    name="scenario_source",
                    value=f"{source.identifier}@{source.version}",
                ),
                AggregationKeyV1(name="scenario_digest", value=source.canonical_digest),
                AggregationKeyV1(
                    name="map_digest", value=specification.layout.canonical_digest
                ),
                AggregationKeyV1(
                    name="scenario_specification_digest",
                    value=specification.canonical_digest_sha256,
                ),
            )
        )
    aggregation_keys.sort(key=lambda row: row.name)

    return build_evaluation_episode_context_v4(
        identity=identity,
        scenario_name=scenario.name,
        aggregation_keys=tuple(aggregation_keys),
        expected_horizon=horizon,
        config=config,
        public_agent_id_by_global_slot=(
            DEBUGGER_PUBLIC_AGENT_IDS_V1
            if specification is None
            else tuple(row.public_agent_id for row in specification.roster_template)
        ),
        policy_assignments=assignments,
        seed_protocol=seed_protocol,
        capture_profile=launch.capture_profile,
        execution_information_mode=execution_information_mode,
        actor_projection=actor_projection,
        critic_information_regime=VersionedIdentityV1(
            identifier="not_applicable",
            version=1,
        ),
        canonical_reward_mode=VersionedIdentityV1(
            identifier="canonical-task-reward",
            version=1,
        ),
        shaping_configuration=_content_identity(
            "not_applicable-no-shaping",
            {
                "schema_id": "marl_battlegrounds.visual_debugger.shaping",
                "schema_version": 1,
                "enabled": False,
            },
        ),
        code_revision=launch.code_revision,
    )


__all__ = [
    "DEBUGGER_EVALUATION_BRIDGE_SCHEMA_VERSION",
    "DEBUGGER_EVALUATION_LAUNCH_SPECIFICATION_SCHEMA_ID",
    "DEBUGGER_PUBLIC_AGENT_IDS_V1",
    "DebuggerActionSourceKindV1",
    "DebuggerCaptureProfileV1",
    "DebuggerEvaluationLaunchSpecificationV1",
    "build_debugger_evaluation_context_v1",
    "build_debugger_evaluation_launch_specification_v1",
    "debugger_action_source_kind_v1",
]
