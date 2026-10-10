"""Define immutable host state for live debugger controls and scripted playback.

The control layer creates ``DebuggerSession`` endpoints and replaces them after
validated actions. These models keep one coherent simulator state, observation,
mask, random key, and evaluation record, plus a short tail of public team
scores for the scorebug's Momentum switch (``DebuggerSession.recent_scores``)
, every view's Champion Streaks counts and Fallen panel history
(``DebuggerSession.champion_takedowns``) and every view's Herald Cards
history (``DebuggerSession.match_moments``).
Pending actions remain separate from
accepted simulator actions. Construction checks host contracts; this module has
no file, network, or simulator-step entry point.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from math import isfinite
from typing import TYPE_CHECKING, Literal, TypeGuard, cast

from marl_battlegrounds.evaluation._replays.wire_shapes import (
    MAX_AGENT_SLOTS_V1 as MAX_AGENT_SLOTS,
)
from marl_battlegrounds.evaluation._replays.wire_shapes import (
    NUM_MOVE_ACTIONS_V1 as NUM_MOVE_ACTIONS,
)
from marl_battlegrounds.evaluation._replays.wire_shapes import (
    NUM_TARGET_ACTIONS_V1 as NUM_TARGET_ACTIONS,
)

if TYPE_CHECKING:
    from jax import Array

    from marl_battlegrounds._tdm_assets import TDMMapInfo
    from marl_battlegrounds.core.types import (
        ActionMask,
        EnvConfig,
        EnvState,
        Observation,
    )
    from marl_battlegrounds.environment import EnvironmentState
    from marl_battlegrounds.evaluation._metrics.metrics import (
        EvaluationTransitionViewV1,
    )
    from marl_battlegrounds.evaluation.models import (
        EvaluationEpisodeContextV4,
        EvaluationFrameV3,
    )
    from marl_battlegrounds.evaluation.policy_execution import System, SystemState
    from marl_battlegrounds.evaluation.scenario import ResolvedScenarioSpecificationV3
    from marl_battlegrounds.rendering.scene import StatusSourceEvidenceStateV2
    from marl_battlegrounds.viewer.champion_streaks import ChampionTakedownsV1
    from marl_battlegrounds.viewer.match_moments import MatchMomentHistoryV1

SCENARIO_OPPONENT_ERROR = (
    "Scenario opponents are fixed. Remove team_b. "
    "Only Team A controllers can be chosen."
)

_MOVE_STAY_V1 = 0

type SystemRegistrations = tuple[
    tuple[str, dict[str, object]], tuple[str, dict[str, object]]
]
type Lane = Literal[0, 1]
type ArmOrigin = Literal["automatic", "explicit"]
type ScenarioMode = Literal["interactive", "scripted"]
type ScenarioAudience = Literal["researcher", "stress"]
type SubmissionKind = Literal["interactive", "scripted"]
# Built-in names or a bounded alias declared by the launcher. Never an import path.
type TeamController = str
# Both teams accept the same controllers; the name is kept for existing callers.
type TeamBController = TeamController
type TeamControllerActionSource = Literal["manual", "scripted", "mixed", "policy"]
type ScenarioSourceKind = Literal[
    "current_buffer",
    "saved_draft",
]

SUPPORTED_TEAM_CONTROLLERS: tuple[TeamController, ...] = (
    "manual",
    "reactive_tdm",
    "random_valid",
    "scenario_5",
    "tdm_gamma",
)
SUPPORTED_TEAM_B_CONTROLLERS: tuple[TeamBController, ...] = SUPPORTED_TEAM_CONTROLLERS
BUILT_IN_CONTROLLER_NAMES: dict[str, str] = {
    "reactive_tdm": "tdm-alpha",
    "random_valid": "random",
    "scenario_5": "tdm-beta",
    "tdm_gamma": "tdm-gamma",
}
"""Each built-in controller kind's fixed public name, as every client shows it.

Keys are the private live controller kinds; values are the names researchers
load (``marl_bgs.load_method("tdm-beta")``). Slot labels and recorded replay
names use the values, never the private kinds. Manual has no entry: its label
is always "Manual".
"""


def is_system_controller(value: object) -> TypeGuard[str]:
    """Tell whether value is a bounded System alias, never a factory path.

    Accept system: followed by 1 to 64 lowercase letters, digits, underscores or
    hyphens. The service separately checks membership in its launch declarations.
    """
    return (
        isinstance(value, str)
        and re.fullmatch(r"system:[a-z0-9_-]{1,64}", value) is not None
    )


def is_team_controller(value: object) -> TypeGuard[str]:
    """Accept an existing built-in choice or a correctly shaped System alias."""
    return isinstance(value, str) and (
        value in SUPPORTED_TEAM_CONTROLLERS or is_system_controller(value)
    )


def team_controller_action_source(
    team_a_controller: TeamController,
    team_b_controller: TeamBController,
    *,
    manual_slots: tuple[int, ...] | None = None,
) -> TeamControllerActionSource:
    """Describe whether the two team controllers supply manual or policy actions.

    Parameters
    ----------
    team_a_controller : TeamController
        Team A's selected controller kind.
    team_b_controller : TeamBController
        Team B's selected controller kind; both teams accept ALPHA
        (``reactive_tdm``), BETA (``scenario_5``) and GAMMA (``tdm_gamma``).

    manual_slots : tuple[int, ...] or None
        Configured physical Manual rows. None keeps whole-team classification.

    Returns
    -------
    {"manual", "policy", "mixed"}
        Manual if both teams are manual, policy if neither is manual, otherwise mixed.
        The classification does not replace the recorded identities of either team.
    """
    if team_a_controller == team_b_controller == "manual":
        return "manual"
    if manual_slots:
        return "mixed"
    if team_a_controller != "manual" and team_b_controller != "manual":
        return "policy"
    return "mixed"


def normalize_slot_ownership(
    *,
    active_slots: tuple[int, ...],
    controllers: tuple[TeamController, TeamController],
    manual_slots: tuple[int, ...] | None,
    slot_labels: tuple[str | None, ...] | None,
    scripted: bool = False,
    system_slot_names: tuple[tuple[str, ...] | None, tuple[str, ...] | None]
    | None = None,
) -> tuple[tuple[int, ...], tuple[str | None, ...]]:
    """Check physical controller ownership and fill legacy whole-team defaults.

    Host-only. It reads plain tuples, calls no method and changes nothing.

    Parameters
    ----------
    active_slots : tuple of int
        Configured physical rows (0 to 9), including dead actors.
    controllers : tuple of two TeamController
        Team A and Team B controller kinds or System aliases, in that order.
    manual_slots : tuple of int or None
        Rows a human plays. None derives them: every active row of a Manual
        team. An explicit tuple may also add Manual rows inside a composed
        System team. Rows must be distinct Python ints. Manual teams own every
        active row and built-in policy teams own none.
    slot_labels : tuple of str or None, or None
        Ten controller labels, one per physical row: nonempty strings for
        active rows and None only for inactive rows. Manual rows must read
        "Manual". None derives the labels: "Manual" for Manual rows, a
        built-in kind's fixed public name (BUILT_IN_CONTROLLER_NAMES, so
        ``tdm_gamma`` reads ``tdm-gamma``), and for a System alias the name
        from ``system_slot_names`` when given, else the alias itself.
    scripted : bool, optional
        True for scripted playback. It has no Manual rows and derived labels
        read "Scripted". Default False.
    system_slot_names : tuple of two (tuple of five str or None), or None
        Optional names for derived labels, in Team A/B order. Each entry gives
        a System-alias team's five names in team-local slot order (slot 0
        first), as ``recording_identity.team_controller_names`` returns them
        for the team's loaded System, or None for a team without names. Only
        derived labels use it, and only for non-Manual rows of a System-alias
        team. Explicit ``slot_labels`` and scripted playback ignore it.
        Default None: a System alias labels its rows with the alias.

    Returns
    -------
    manual_slots : tuple of int
        The Manual rows, sorted.
    slot_labels : tuple of str or None
        The ten checked or derived labels, as an immutable tuple.

    Raises
    ------
    ValueError
        If a Manual row is not a configured row given as a distinct Python
        int; if scripted playback has Manual rows; if a Manual team's row is
        not Manual; if a built-in policy team owns a Manual row; or if the
        labels are not ten rows with nonempty text on exactly the active rows
        and "Manual" on Manual rows. The caller runs this before any method
        is initialized or called.
    """
    if manual_slots is None:
        manual_slots = tuple(
            slot
            for slot in active_slots
            if not scripted and controllers[slot // 5] == "manual"
        )
    if type(manual_slots) is not tuple or any(
        type(slot) is not int or slot not in active_slots for slot in manual_slots
    ):
        raise ValueError("Manual slots must be configured physical row integers")
    if len(manual_slots) != len(set(manual_slots)):
        raise ValueError("Manual slots must not contain duplicates")
    ordered = tuple(sorted(manual_slots))
    if ordered != manual_slots:
        manual_slots = ordered
    if scripted and manual_slots:
        raise ValueError("Scripted playback cannot own Manual rows")
    for slot in active_slots:
        controller = controllers[slot // 5]
        if not scripted and controller == "manual" and slot not in manual_slots:
            raise ValueError("Every configured Manual team row must be Manual")
        if (
            slot in manual_slots
            and controller != "manual"
            and not is_system_controller(controller)
        ):
            raise ValueError(
                "Only a Manual or composed System team can own Manual rows"
            )
    if slot_labels is None:
        names = (None, None) if system_slot_names is None else system_slot_names
        derived: list[str | None] = []
        for slot in range(MAX_AGENT_SLOTS):
            controller = controllers[slot // 5]
            team_names = names[slot // 5]
            if slot not in active_slots:
                derived.append(None)
            elif scripted:
                derived.append("Scripted")
            elif slot in manual_slots:
                derived.append("Manual")
            elif is_system_controller(controller) and team_names is not None:
                derived.append(team_names[slot % 5])
            else:
                derived.append(BUILT_IN_CONTROLLER_NAMES.get(controller, controller))
        slot_labels = tuple(derived)
    if type(slot_labels) is not tuple or len(slot_labels) != MAX_AGENT_SLOTS:
        raise ValueError("Controller labels must contain ten physical rows")
    for slot, label in enumerate(slot_labels):
        if slot not in active_slots:
            if label is not None:
                raise ValueError("Inactive rows must not have a controller label")
        elif not isinstance(label, str) or not label.strip():
            raise ValueError("Configured rows need a nonempty controller label")
        elif slot in manual_slots and label != "Manual":
            raise ValueError("Manual rows must have the Manual label")
    return manual_slots, slot_labels


def _validate_slot(global_slot: int, *, name: str) -> None:
    """Reject a global agent slot outside the fixed ten-slot axis."""
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        msg = f"{name} must be in [0, {MAX_AGENT_SLOTS}); got {global_slot}."
        raise ValueError(msg)


def _validate_recent_scores(
    recent_scores: object,
    frame_index: int,
    scores: tuple[int, ...],
) -> None:
    """Check DebuggerSession.recent_scores against the current frame.

    recent_scores must be a tuple of (Team A, Team B) pairs of Python ints
    0 or more, at most MOMENTUM_TICKS (10) long and no longer than
    frame_index (the current frame's index, so frame zero holds none). A
    team's score never goes down: each pair is at most the next one, and the
    newest at most scores (the current frame's two scores). Raises TypeError
    when it is not a tuple of pairs of nonnegative Python ints, and
    ValueError for a wrong length or a falling score.
    A shorter tail than possible is allowed; it only shows fewer recent
    points.
    """
    from marl_battlegrounds.viewer.match_summary import MOMENTUM_TICKS

    shape_error = "recent_scores must hold pairs of nonnegative Python ints."
    if type(recent_scores) is not tuple:
        raise TypeError(shape_error)
    for pair in cast("tuple[object, ...]", recent_scores):
        if type(pair) is not tuple:
            raise TypeError(shape_error)
        values = cast("tuple[object, ...]", pair)
        if len(values) != 2 or any(
            type(value) is not int or value < 0 for value in values
        ):
            raise TypeError(shape_error)
    pairs = cast("tuple[tuple[int, int], ...]", recent_scores)
    if len(pairs) > min(MOMENTUM_TICKS, frame_index):
        raise ValueError(
            "recent_scores cannot hold more frames than came before this one."
        )
    current = (int(scores[0]), int(scores[1]))
    for earlier, later in pairwise((*pairs, current)):
        if earlier[0] > later[0] or earlier[1] > later[1]:
            raise ValueError("recent_scores cannot show a team's score going down.")


def _no_champion_takedowns() -> ChampionTakedownsV1:
    """Return the zero Champion Streaks record of a new game (no takedowns yet).

    DebuggerSession.champion_takedowns uses it as its default. The import stays
    inside so importing this module loads no rendering or evaluation records.
    """
    from marl_battlegrounds.viewer.champion_streaks import ChampionTakedownsV1

    return ChampionTakedownsV1()


def _no_match_moments() -> MatchMomentHistoryV1:
    """Return the empty Herald Cards history of a new game (no tick played yet).

    DebuggerSession.match_moments uses it as its default. The import stays
    inside so importing this module loads no rendering or evaluation records.
    """
    from marl_battlegrounds.viewer.match_moments import MatchMomentHistoryV1

    return MatchMomentHistoryV1()


@dataclass(frozen=True, slots=True)
class PendingAction:
    """One actor's editable move, target, and selected Basic/Ultimate lane.

    ``armed_lane`` is 0 for Basic, 1 for Ultimate, or None when neither is armed.
    ``arm_origin`` records whether that choice was automatic or explicit.
    """

    move_action: int = _MOVE_STAY_V1
    selected_global_target_slot: int | None = None
    armed_lane: Lane | None = 0
    arm_origin: ArmOrigin | None = "automatic"

    def __post_init__(self) -> None:
        """Validate action ranges and require an arm origin exactly when a lane is
        armed.
        """
        if not 0 <= self.move_action < NUM_MOVE_ACTIONS:
            msg = (
                f"move_action must be in [0, {NUM_MOVE_ACTIONS}); "
                f"got {self.move_action}."
            )
            raise ValueError(msg)
        if self.selected_global_target_slot is not None:
            _validate_slot(
                self.selected_global_target_slot,
                name="selected_global_target_slot",
            )
        if self.armed_lane not in (None, 0, 1):
            msg = f"armed_lane must be None, 0, or 1; got {self.armed_lane}."
            raise ValueError(msg)
        if self.arm_origin not in (None, "automatic", "explicit"):
            msg = f"unknown arm_origin: {self.arm_origin!r}."
            raise ValueError(msg)
        if (self.armed_lane is None) != (self.arm_origin is None):
            msg = "arm_origin must be None exactly when armed_lane is None."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LaneAvailability:
    """Legality of both Basic/Ultimate lanes for one target choice.

    ``armed_pair_legal`` must equal the availability of the selected lane; no
    selected lane means that pair is not ready to submit.
    """

    target_action: int
    lane_0_available: bool
    lane_1_available: bool
    armed_lane: Lane | None
    armed_pair_legal: bool

    def __post_init__(self) -> None:
        """Require valid target/lane values and consistent selected-pair legality."""
        if not 0 <= self.target_action < NUM_TARGET_ACTIONS:
            msg = f"invalid target_action: {self.target_action}."
            raise ValueError(msg)
        if self.armed_lane not in (None, 0, 1):
            msg = f"armed_lane must be None, 0, or 1; got {self.armed_lane}."
            raise ValueError(msg)
        expected = (
            False
            if self.armed_lane is None
            else self.lane_0_available
            if self.armed_lane == 0
            else self.lane_1_available
        )
        if self.armed_pair_legal != expected:
            msg = "armed_pair_legal must match the selected exact lane value."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ActorCommand:
    """One scripted actor command using global actor and target slots.

    Missing targets mean no target. ``use_ultimate`` is 0 for Basic or 1 for Ultimate;
    legality against current masks is checked when the command is submitted.
    """

    actor_global_slot: int
    move_action: int = _MOVE_STAY_V1
    target_global_slot: int | None = None
    use_ultimate: int = 0

    def __post_init__(self) -> None:
        """Check fixed action categories and actor/target slot ranges."""
        _validate_slot(self.actor_global_slot, name="actor_global_slot")
        if not 0 <= self.move_action < NUM_MOVE_ACTIONS:
            msg = (
                f"move_action must be in [0, {NUM_MOVE_ACTIONS}); "
                f"got {self.move_action}."
            )
            raise ValueError(msg)
        if self.target_global_slot is not None:
            _validate_slot(self.target_global_slot, name="target_global_slot")
        if self.use_ultimate not in (0, 1):
            msg = f"use_ultimate must be 0 or 1; got {self.use_ultimate}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ScenarioFrame:
    """One labeled scripted transition containing at most one command per actor."""

    label: str
    description: str
    commands: tuple[ActorCommand, ...]

    def __post_init__(self) -> None:
        """Reject duplicate commands for the same actor in one scripted frame."""
        actor_slots = tuple(command.actor_global_slot for command in self.commands)
        if len(actor_slots) != len(set(actor_slots)):
            msg = f"scenario frame {self.label!r} contains duplicate actor commands."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DebuggerScenarioProvenance:
    """Exact authored-source identities retained by diagnostic captures."""

    source_kind: ScenarioSourceKind
    source_identity: str
    scenario_semantic_digest: str
    map_semantic_digest: str
    resolved_configuration_digest: str
    resolved_initial_state_digest: str
    map_id: int | None = None

    def __post_init__(self) -> None:
        """Require a known source kind, valid optional map ID, and exact SHA-256
        digests.
        """
        if self.source_kind not in ("current_buffer", "saved_draft"):
            raise ValueError("unknown authored scenario source kind.")
        if not self.source_identity:
            raise ValueError("source_identity must be nonempty.")
        if self.map_id is not None and (
            type(self.map_id) is not int or not 0 <= self.map_id <= 51
        ):
            raise ValueError("map_id must identify an approved TDM map.")
        for name, value in (
            ("scenario_semantic_digest", self.scenario_semantic_digest),
            ("map_semantic_digest", self.map_semantic_digest),
            ("resolved_configuration_digest", self.resolved_configuration_digest),
            ("resolved_initial_state_digest", self.resolved_initial_state_digest),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex.")


@dataclass(frozen=True, slots=True)
class DebuggerScenario:
    """A live or scripted scenario factory and its ordered transition instructions.

    The factory returns fresh configuration and state. Audience distinguishes
    researcher examples from explicitly selected stress demonstrations.
    scenario_specification retains one exact packaged scenario's setup; packaged_map
    retains an ordinary map's catalog identity. Both default to None and are
    mutually exclusive with each other and authored provenance.

    spawn_banks_exchanged (default False) says that the factory's map is
    packaged_map with its two spawn banks exchanged, so Team A starts at the
    end Team B starts at in the catalog (``tasks._swap_spawn_banks``). Only a
    packaged map may declare it. The evaluation bridge then checks the map
    against the catalog with the banks put back, names it as that map and
    records the exchange (``marl_bgs.spawn_locations`` "1"); if that check
    fails, the bridge records a custom map instead. False keeps the exact
    bank-order check every packaged map uses and records
    ``marl_bgs.spawn_locations`` "0".
    """

    name: str
    title: str
    description: str
    mode: ScenarioMode
    build_scenario: Callable[[], tuple[EnvConfig, EnvState]]
    frames: tuple[ScenarioFrame, ...]
    default_controlled_slot: int
    audience: ScenarioAudience = "researcher"
    provenance: DebuggerScenarioProvenance | None = None
    scenario_specification: ResolvedScenarioSpecificationV3 | None = None
    packaged_map: TDMMapInfo | None = None
    spawn_banks_exchanged: bool = False

    def __post_init__(self) -> None:
        """Validate the controlled slot, scenario mode, audience and map source.

        Raises
        ------
        ValueError
            More than one content source is given, a fixed scenario is invalid,
            spawn_banks_exchanged is True without a packaged map, or the slot,
            mode or audience is unknown.
        TypeError
            A fixed scenario or packaged map is not its exact catalog record,
            or spawn_banks_exchanged is not a bool.
        """
        if (
            sum(
                value is not None
                for value in (
                    self.provenance,
                    self.scenario_specification,
                    self.packaged_map,
                )
            )
            > 1
        ):
            raise ValueError("A scenario can have only one content source")
        if self.scenario_specification is not None:
            from marl_battlegrounds.evaluation.scenario import (
                ResolvedScenarioSpecificationV3,
            )

            if type(self.scenario_specification) is not ResolvedScenarioSpecificationV3:
                raise TypeError("Fixed scenarios require the existing V3 specification")
            if (
                re.fullmatch(
                    r"tdm-scenario-[1-8]", self.scenario_specification.scenario_id
                )
                is None
            ):
                raise ValueError(
                    "Fixed scenarios must name an installed scenario from 1 through 8"
                )
            if self.mode != "interactive":
                raise ValueError("Fixed scenarios use interactive controls")
        if self.packaged_map is not None:
            from marl_battlegrounds._tdm_assets import TDMMapInfo

            if type(self.packaged_map) is not TDMMapInfo:
                raise TypeError("Packaged maps require their installed catalog entry")
        if type(self.spawn_banks_exchanged) is not bool:
            raise TypeError("spawn_banks_exchanged must be True or False")
        if self.spawn_banks_exchanged and self.packaged_map is None:
            raise ValueError("Only a packaged map can declare exchanged spawn banks")
        _validate_slot(self.default_controlled_slot, name="default_controlled_slot")
        if self.mode not in ("interactive", "scripted"):
            msg = f"unknown scenario mode: {self.mode!r}."
            raise ValueError(msg)
        if self.audience not in ("researcher", "stress"):
            msg = f"unknown scenario audience: {self.audience!r}."
            raise ValueError(msg)

    @property
    def packaged_scenario_id(self) -> int | None:
        """Return the fixed packaged scenario number, or None for other starts."""
        specification = self.scenario_specification
        return (
            None
            if specification is None
            else int(specification.scenario_id.rsplit("-", 1)[1])
        )

    @property
    def numeric_episode_id(self) -> int:
        """Return first qualification case ID, or 1 for ordinary diagnostics.

        Fixed scenarios use coordinate zero of the existing two-case schedule.
        The ID stays the same across restarts; request generations are separate.
        """
        identifier = self.packaged_scenario_id
        return 1 if identifier is None else (identifier - 1) * 2 + 1


@dataclass(frozen=True, slots=True)
class RawContinuationIdentity:
    """Identity-bind the exact raw objects that feed the next core step."""

    config: EnvConfig
    key: Array
    state: EnvState
    observation: Observation
    action_mask: ActionMask
    systems: tuple[System, System] | None = None
    environment_state: EnvironmentState | None = None
    system_memory: SystemState | None = None
    system_ids: tuple[str, str] | None = None
    system_registrations: SystemRegistrations | None = None
    numeric_episode_id: int = 1
    borrow_systems: bool = False
    manual_slots: tuple[int, ...] = ()
    slot_labels: tuple[str | None, ...] = ()
    scenario: DebuggerScenario | None = None
    evaluation_context: EvaluationEpisodeContextV4 | None = None
    local_step: int = 0

    def matches(
        self,
        *,
        config: EnvConfig,
        key: Array,
        state: EnvState,
        observation: Observation,
        action_mask: ActionMask,
        systems: tuple[System, System] | None = None,
        environment_state: EnvironmentState | None = None,
        system_memory: SystemState | None = None,
        system_ids: tuple[str, str] | None = None,
        system_registrations: SystemRegistrations | None = None,
        numeric_episode_id: int = 1,
        borrow_systems: bool = False,
        manual_slots: tuple[int, ...] = (),
        slot_labels: tuple[str | None, ...] = (),
        scenario: DebuggerScenario | None = None,
        evaluation_context: EvaluationEpisodeContextV4 | None = None,
        local_step: int = 0,
    ) -> bool:
        """Check that a continuation still uses the exact objects that were bound.

        Parameters
        ----------
        config : EnvConfig
            Configuration to use for the next transition.
        key : Array
            Random key to use for the next transition.
        state : EnvState
            Current simulator state.
        observation : Observation
            Observation produced with that state.
        action_mask : ActionMask
            Action mask produced with that state.
        systems, environment_state, system_memory, system_ids : optional
            Exact declared methods, wrapper, accepted memory and setup IDs.
            All are None for the original built-in route.
        system_registrations : tuple or None
            Recorded method facts from setup, reused without reading weights again.
        numeric_episode_id, local_step : int
            Numeric System episode ID and local decision index, default 1 and 0.
        borrow_systems : bool
            Whether supplied method ownership stays with the caller, default False.
        manual_slots, slot_labels : tuple
            Exact configured Manual rows and ten physical display labels.
        scenario, evaluation_context : optional
            Exact setup and seed records. None supports standalone old identities.

        Returns
        -------
        bool
            True only when objects retain identity and numeric coordinates match.
            This does not compare array contents or copy device buffers.
        """
        return (
            self.config is config
            and self.key is key
            and self.state is state
            and self.observation is observation
            and self.action_mask is action_mask
            and self.systems is systems
            and self.environment_state is environment_state
            and self.system_memory is system_memory
            and self.system_ids is system_ids
            and self.system_registrations is system_registrations
            and self.numeric_episode_id == numeric_episode_id
            and self.borrow_systems == borrow_systems
            and self.manual_slots is manual_slots
            and self.slot_labels is slot_labels
            and self.scenario is scenario
            and self.evaluation_context is evaluation_context
            and self.local_step == local_step
        )


@dataclass(frozen=True, slots=True)
class DebuggerSession:
    """One immutable live endpoint with its raw continuation and recording context.

    State, observation, mask, config, and random key must refer to the same next
    simulator step. Host controls and pending actions live beside that endpoint.
    systems, environment_state and system_memory are all absent for built-ins.
    A declared System route retains both methods, a matching public wrapper state
    and only the last accepted memory. Resource and worker ownership stays with
    the service, outside this numerical snapshot. borrow_systems=False keeps the
    legacy method snapshot; True retains caller-owned methods and their parameters
    through restarts. Callers must keep those parameters fixed during the launch.
    system_registrations retains their setup facts so evidence and restart do not
    read large weights again. These host-owned facts never enter actor inputs.
    manual_slots holds configured human-owned physical rows, including dead actors.
    slot_labels has ten entries; inactive rows alone use None. Focus never changes
    ownership. Both tuples are fixed for an episode and bound to its continuation.
    numeric_episode_id defaults to
    1 for diagnostics; fixed scenarios retain their first qualification case ID.
    It is separate from the string recording ID and restart request generation.
    recent_scores is the live service's short score history for the Momentum
    switch (hud_momentum): the [Team A, Team B] scores of the frames before
    the current one, oldest first, at most viewer.match_summary.MOMENTUM_TICKS
    (10) of them and never more than the current frame index. It is empty on
    frame zero (a new game, Reset or restart). Each transition appends the
    old frame's scores (next_recent_scores); scores_ten_ticks_ago reads it.
    Scores are public match facts; they never enter actor inputs.
    champion_takedowns holds every view's counts of takedowns without falling
    after the current frame, for the Champion Streaks switch
    (viewer.champion_streaks.ChampionTakedownsV1), and each view's recent
    ticks with a fall, for the Fallen panel's earlier rows (recent_falls):
    Oracle View and the Agent POV of every possible recipient, so a view
    switch is exact at once. It is
    the zero record on frame zero (a new game, Reset or restart); each
    transition advances it (live_presentation.
    advance_live_champion_takedowns_v1). These are display facts only; they
    never enter actor inputs.
    match_moments holds every view's Herald Cards history after the current
    frame (viewer.match_moments.MatchMomentHistoryV1): whether the match
    began at tick 0 with 0-0, the largest lead at the start of any tick so
    far, and each view's unanswered runs and this tick's rescues, for Oracle
    View and the Agent POV of every possible recipient. It is the empty
    record on frame zero (a new game, Reset or restart); each transition
    advances it together with champion_takedowns
    (live_presentation.advance_live_match_history_v1). Display facts only;
    they never enter actor inputs.
    """

    scenario: DebuggerScenario
    seed: int
    run_generation: int
    scenario_default_movement_scale: float
    config: EnvConfig
    key: Array
    state: EnvState
    observation: Observation
    action_mask: ActionMask
    evaluation_context: EvaluationEpisodeContextV4
    current_evaluation_frame: EvaluationFrameV3
    incoming_evaluation_view: EvaluationTransitionViewV1 | None
    status_source_evidence_state: StatusSourceEvidenceStateV2
    last_submission_kind: SubmissionKind | None
    last_report_actor_slots: tuple[int, ...]
    team_a_controller: TeamController
    team_b_controller: TeamBController
    controlled_global_slot: int
    pending_actions: tuple[PendingAction, ...]
    next_script_frame_index: int
    show_ranges: bool
    verbose_logging: bool
    raw_continuation_identity: RawContinuationIdentity | None = None
    systems: tuple[System, System] | None = None
    environment_state: EnvironmentState | None = None
    system_memory: SystemState | None = None
    system_ids: tuple[str, str] | None = None
    system_registrations: SystemRegistrations | None = None
    numeric_episode_id: int = 1
    borrow_systems: bool = False
    manual_slots: tuple[int, ...] = ()
    slot_labels: tuple[str | None, ...] = ()
    recent_scores: tuple[tuple[int, int], ...] = ()
    champion_takedowns: ChampionTakedownsV1 = field(
        default_factory=_no_champion_takedowns
    )
    match_moments: MatchMomentHistoryV1 = field(default_factory=_no_match_moments)

    @property
    def scenario_name(self) -> str:
        """Return the owned scenario's stable display and provenance name."""
        return self.scenario.name

    @property
    def scores_ten_ticks_ago(self) -> tuple[int, int]:
        """Return each team's score 10 ticks before the current frame.

        The result is (Team A, Team B): the oldest entry of recent_scores,
        which is the frame 10 frames back, or frame zero in a younger game;
        on frame zero (empty recent_scores) the current frame's own scores.
        The live presentation passes it to
        viewer.match_summary.build_match_summary_v1 for the Momentum switch.
        Pure; reads this session only.
        """
        if self.recent_scores:
            return self.recent_scores[0]
        scores = self.current_evaluation_frame.snapshot.team_deathmatch_scores
        return (int(scores[0]), int(scores[1]))

    def next_recent_scores(self) -> tuple[tuple[int, int], ...]:
        """Return recent_scores for the session one transition later.

        It is this session's recent_scores with the current frame's
        (Team A, Team B) scores appended, keeping only the newest
        viewer.match_summary.MOMENTUM_TICKS (10). control.submit_joint_action
        stores it on the successor session; a new game or restart starts
        from an empty tuple instead. Pure; returns a new tuple.
        """
        from marl_battlegrounds.viewer.match_summary import MOMENTUM_TICKS

        scores = self.current_evaluation_frame.snapshot.team_deathmatch_scores
        return (*self.recent_scores, (int(scores[0]), int(scores[1])))[-MOMENTUM_TICKS:]

    def __post_init__(self) -> None:
        """Bind the exact continuation objects and validate the host endpoint
        contract.
        """
        from marl_battlegrounds.evaluation._metrics.metrics import (
            EvaluationTransitionViewV1,
        )
        from marl_battlegrounds.evaluation.models import (
            AssignedPolicySlotV2,
            EvaluationEpisodeContextV4,
            EvaluationFrameV3,
        )
        from marl_battlegrounds.rendering.scene import StatusSourceEvidenceStateV2

        if type(self.scenario) is not DebuggerScenario:
            raise TypeError("scenario must be the exact DebuggerScenario type.")
        if type(self.borrow_systems) is not bool:
            raise TypeError("borrow_systems must be a Python bool")
        fixed_scenario = (
            self.scenario if self.scenario.scenario_specification is not None else None
        )
        key_context = self.evaluation_context if fixed_scenario is not None else None
        local_step = (
            self.current_evaluation_frame.frame_index
            if fixed_scenario is not None
            else 0
        )
        continuation = self.raw_continuation_identity
        new_continuation = continuation is None
        if continuation is None:
            continuation = RawContinuationIdentity(
                config=self.config,
                key=self.key,
                state=self.state,
                observation=self.observation,
                action_mask=self.action_mask,
                systems=self.systems,
                environment_state=self.environment_state,
                system_memory=self.system_memory,
                system_ids=self.system_ids,
                system_registrations=self.system_registrations,
                numeric_episode_id=self.numeric_episode_id,
                borrow_systems=self.borrow_systems,
                manual_slots=self.manual_slots,
                slot_labels=self.slot_labels,
                scenario=fixed_scenario,
                evaluation_context=key_context,
                local_step=local_step,
            )
            object.__setattr__(self, "raw_continuation_identity", continuation)
        elif type(
            continuation
        ) is not RawContinuationIdentity or not continuation.matches(
            config=self.config,
            key=self.key,
            state=self.state,
            observation=self.observation,
            action_mask=self.action_mask,
            systems=self.systems,
            environment_state=self.environment_state,
            system_memory=self.system_memory,
            system_ids=self.system_ids,
            system_registrations=self.system_registrations,
            numeric_episode_id=self.numeric_episode_id,
            borrow_systems=self.borrow_systems,
            manual_slots=self.manual_slots,
            slot_labels=self.slot_labels,
            scenario=fixed_scenario,
            evaluation_context=key_context,
            local_step=local_step,
        ):
            raise ValueError(
                "raw continuation identity must bind the exact next-step objects."
            )
        if (
            type(self.numeric_episode_id) is not int
            or self.numeric_episode_id != self.scenario.numeric_episode_id
        ):
            raise ValueError(
                "Numeric episode ID must match the scenario's first seed case"
            )
        if type(self.run_generation) is not int or self.run_generation < 0:
            msg = "run_generation must be a non-negative Python int."
            raise ValueError(msg)
        if type(self.evaluation_context) is not EvaluationEpisodeContextV4:
            raise TypeError(
                "evaluation_context must be the exact EvaluationEpisodeContextV4 root."
            )
        if type(self.current_evaluation_frame) is not EvaluationFrameV3:
            raise TypeError(
                "current_evaluation_frame must be the exact EvaluationFrameV3 root."
            )
        if (
            type(self.scenario_default_movement_scale) is not float
            or not isfinite(self.scenario_default_movement_scale)
            or not 0.0 < self.scenario_default_movement_scale <= 1.0
        ):
            msg = (
                "scenario_default_movement_scale must be a finite Python float "
                "in (0.0, 1.0]."
            )
            raise ValueError(msg)
        _validate_slot(self.controlled_global_slot, name="controlled_global_slot")
        roster = self.evaluation_context.roster
        if not roster[self.controlled_global_slot].configured_active:
            msg = f"controlled slot g{self.controlled_global_slot} is inactive."
            raise ValueError(msg)
        if len(self.pending_actions) != MAX_AGENT_SLOTS:
            msg = (
                f"pending_actions must contain {MAX_AGENT_SLOTS} fixed-slot rows; "
                f"got {len(self.pending_actions)}."
            )
            raise ValueError(msg)
        inactive_pending = PendingAction(armed_lane=None, arm_origin=None)
        runtime_pending_actions = cast(tuple[object, ...], self.pending_actions)
        for actor_slot, pending in enumerate(runtime_pending_actions):
            if not isinstance(pending, PendingAction):
                msg = f"pending_actions[{actor_slot}] must be a PendingAction."
                raise TypeError(msg)
            actor_active = roster[actor_slot].configured_active
            if not actor_active and pending != inactive_pending:
                msg = f"inactive pending row g{actor_slot} must remain neutral."
                raise ValueError(msg)
            target_slot = pending.selected_global_target_slot
            if target_slot is not None and not roster[target_slot].configured_active:
                msg = f"pending target g{target_slot} is inactive."
                raise ValueError(msg)
        if self.next_script_frame_index < 0:
            msg = "next_script_frame_index must be non-negative."
            raise ValueError(msg)
        episode_id = self.evaluation_context.identity.episode_id
        current_frame = self.current_evaluation_frame
        if current_frame.episode_id != episode_id:
            raise ValueError("current evaluation frame must join the session episode.")
        if current_frame.frame_index > self.evaluation_context.expected_horizon:
            raise ValueError(
                "current evaluation frame cannot exceed the declared horizon."
            )
        incoming = self.incoming_evaluation_view
        if incoming is None:
            if current_frame.frame_index != 0:
                raise ValueError(
                    "only evaluation frame zero may omit an incoming transition."
                )
            if self.last_submission_kind is not None or self.last_report_actor_slots:
                raise ValueError(
                    "initial sessions cannot retain transition submission metadata."
                )
        else:
            if type(incoming) is not EvaluationTransitionViewV1:
                raise TypeError(
                    "incoming_evaluation_view must be an exact coherent V1 view."
                )
            if (
                incoming.context != self.evaluation_context
                or incoming.successor_frame != current_frame
                or incoming.transition.transition_index != current_frame.frame_index - 1
            ):
                raise ValueError(
                    "incoming evaluation view must enter the current session frame."
                )
            if self.last_submission_kind not in ("interactive", "scripted"):
                raise ValueError(
                    "non-initial sessions require exact submission-kind metadata."
                )
        _validate_recent_scores(
            self.recent_scores,
            current_frame.frame_index,
            current_frame.snapshot.team_deathmatch_scores,
        )
        from marl_battlegrounds.viewer.champion_streaks import ChampionTakedownsV1

        if type(self.champion_takedowns) is not ChampionTakedownsV1:
            raise TypeError("champion_takedowns must be a ChampionTakedownsV1.")
        if current_frame.frame_index == 0 and (
            self.champion_takedowns != ChampionTakedownsV1()
        ):
            raise ValueError("frame zero cannot hold champion takedowns.")
        from marl_battlegrounds.viewer.match_moments import MatchMomentHistoryV1

        if type(self.match_moments) is not MatchMomentHistoryV1:
            raise TypeError("match_moments must be a MatchMomentHistoryV1.")
        if current_frame.frame_index == 0 and (
            self.match_moments != MatchMomentHistoryV1()
        ):
            raise ValueError("frame zero cannot hold Herald Cards history.")
        if type(self.last_report_actor_slots) is not tuple:
            raise TypeError("last_report_actor_slots must be a Python tuple.")
        if self.last_report_actor_slots != tuple(
            sorted(set(self.last_report_actor_slots))
        ):
            raise ValueError("last_report_actor_slots must be sorted and unique.")
        for actor_slot in self.last_report_actor_slots:
            _validate_slot(actor_slot, name="last_report_actor_slot")
            if not self.evaluation_context.roster[actor_slot].configured_active:
                raise ValueError("last report actor slots must be configured active.")
        if not is_team_controller(self.team_a_controller):
            raise ValueError(
                "team_a_controller must be manual, reactive_tdm, random_valid, "
                "scenario_5, or tdm_gamma."
            )
        if not is_team_controller(self.team_b_controller):
            raise ValueError(
                "team_b_controller must be manual, reactive_tdm, random_valid, "
                "scenario_5, or tdm_gamma."
            )
        if (
            any(
                controller in ("reactive_tdm", "scenario_5", "tdm_gamma")
                or is_system_controller(controller)
                for controller in (self.team_a_controller, self.team_b_controller)
            )
            and self.evaluation_context.execution_information_mode != "shared_obs"
        ):
            raise ValueError("Reactive controllers require SharedObs.")
        if self.scenario.mode == "scripted":
            if self.team_a_controller != "manual" or self.team_b_controller != "manual":
                raise ValueError(
                    "registered scripted scenarios do not use interactive team "
                    "controllers."
                )
        else:
            expected_action_source = team_controller_action_source(
                self.team_a_controller,
                self.team_b_controller,
                manual_slots=self.manual_slots,
            )
            expected_aggregation = {
                "action_source": expected_action_source,
                "team_a_controller": self.team_a_controller,
                "team_b_controller": self.team_b_controller,
            }
            for name, expected in expected_aggregation.items():
                values = tuple(
                    row.value
                    for row in self.evaluation_context.aggregation_keys
                    if row.name == name
                )
                if values != (expected,):
                    raise ValueError(
                        f"session {name} must join its evaluation context."
                    )
            team_a_id = roster[0].configured_team_id
            for slot, roster_row in enumerate(roster):
                if not roster_row.configured_active:
                    continue
                assignment = self.evaluation_context.policy_assignments[slot]
                expected_policy_kind = (
                    "manual"
                    if slot in self.manual_slots
                    else self.team_a_controller
                    if roster_row.configured_team_id == team_a_id
                    else self.team_b_controller
                )
                if is_system_controller(expected_policy_kind):
                    if self.systems is None:
                        raise ValueError("A System controller needs its actual method")
                    team = 0 if roster_row.configured_team_id == team_a_id else 1
                    if self.system_ids is None:
                        raise ValueError("System identities must be bound at setup")
                    from marl_battlegrounds.evaluation._recording.recording_context import (  # noqa: E501
                        recorded_policy_assignment,
                    )

                    if not isinstance(assignment, AssignedPolicySlotV2):
                        raise ValueError(
                            "System assignment must match its actual method"
                        )

                    if self.system_registrations is None:
                        raise ValueError(
                            "System setup records must match their recorded IDs"
                        )
                    identifier, registration = self.system_registrations[team]
                    expected_assignment = recorded_policy_assignment(
                        slot=slot,
                        role=assignment.evaluation_role,
                        phase="evaluation",
                        name=registration["name"],
                        descriptor=registration,
                        system_id=identifier,
                        registration=registration,
                    )
                    if assignment != expected_assignment:
                        raise ValueError(
                            "System assignment must match its actual method"
                        )
                    continue
                if (
                    not isinstance(assignment, AssignedPolicySlotV2)
                    or assignment.policy_kind != expected_policy_kind
                ):
                    raise ValueError(
                        "session team controllers must join every active policy "
                        "assignment."
                    )
        manual_slots, _ = normalize_slot_ownership(
            active_slots=tuple(
                row.global_slot for row in roster if row.configured_active
            ),
            controllers=(self.team_a_controller, self.team_b_controller),
            manual_slots=self.manual_slots,
            slot_labels=self.slot_labels,
            scripted=self.scenario.mode == "scripted",
        )
        if manual_slots != self.manual_slots:
            raise ValueError("Manual slots must be in physical row order")
        system_route = any(
            is_system_controller(value)
            for value in (self.team_a_controller, self.team_b_controller)
        )
        if system_route != (self.systems is not None):
            raise ValueError("System methods must match the selected controllers")
        if self.systems is None:
            if (
                self.environment_state is not None
                or self.system_memory is not None
                or self.system_ids is not None
                or self.system_registrations is not None
            ):
                raise ValueError("A built-in session cannot retain System memory")
        else:
            if (
                self.system_registrations is None
                or tuple(row[0] for row in self.system_registrations) != self.system_ids
            ):
                raise ValueError("System setup records must match their recorded IDs")
            wrapped = self.environment_state
            if wrapped is None or self.system_memory is None:
                raise ValueError("A System session needs paired state and memory")
            if (
                wrapped.config is not self.config
                or wrapped.core_state is not self.state
                or wrapped.observation is not self.observation
                or wrapped.action_mask is not self.action_mask
            ):
                raise ValueError("System state must bind the exact current game view")
            if new_continuation:
                import numpy as np

                captured = current_frame.shared_obs_information_availability_by_recipient_and_sensor_source  # noqa: E501
                if captured is None or not np.array_equal(
                    np.asarray(wrapped.source_availability), captured
                ):
                    raise ValueError(
                        "System source permissions must match the captured frame"
                    )
                if (
                    int(wrapped.episode_id) != self.numeric_episode_id
                    or int(wrapped.reset_generation) != 0
                    or int(wrapped.cumulative_transition_count)
                    != current_frame.frame_index
                    or int(wrapped.initial_step_count)
                    != int(self.state.step_count) - current_frame.frame_index
                    or bool(wrapped.episode_start) != (current_frame.frame_index == 0)
                    or bool(wrapped.lifecycle_error)
                    or bool(wrapped.done.terminated) != self.terminated
                    or bool(wrapped.done.truncated) != self.truncated
                    or not np.all(
                        np.asarray(self.system_memory.episode_id)
                        == self.numeric_episode_id
                    )
                    or not np.all(np.asarray(self.system_memory.reset_generation) == 0)
                ):
                    raise ValueError(
                        "System lifecycle must match the accepted game epoch"
                    )
        evidence_state = self.status_source_evidence_state
        if type(evidence_state) is not StatusSourceEvidenceStateV2:
            raise TypeError("status_source_evidence_state must be the exact V2 state.")
        if (
            evidence_state.episode_id != episode_id
            or evidence_state.frame_index != current_frame.frame_index
            or evidence_state.frame_id != current_frame.frame_id
        ):
            raise ValueError(
                "status-source evidence must join the current evaluation frame."
            )

    @property
    def pending_action(self) -> PendingAction:
        """Return the currently controlled actor's authoritative draft row."""
        return self.pending_actions[self.controlled_global_slot]

    @property
    def terminated(self) -> bool:
        """Return canonical task-termination truth for the current frame."""
        incoming = self.incoming_evaluation_view
        return incoming is not None and incoming.transition.terminated

    @property
    def truncated(self) -> bool:
        """Return canonical truncation truth for the current frame."""
        incoming = self.incoming_evaluation_view
        return incoming is not None and incoming.transition.truncated

    @property
    def reached_declared_horizon(self) -> bool:
        """Return whether the canonical episode prefix reached its exact horizon."""
        return (
            self.current_evaluation_frame.frame_index
            == self.evaluation_context.expected_horizon
        )

    @property
    def episode_sealed(self) -> bool:
        """Return whether another simulator transition is scientifically invalid."""
        return self.terminated or self.truncated or self.reached_declared_horizon
