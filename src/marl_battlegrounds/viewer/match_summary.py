"""Build current-frame match facts for researcher panels and beat timing.

Use ``build_match_summary_v1`` with the recorded context, frame, incoming
events and, when known, the frame the transition started from and each team's
score ``MOMENTUM_TICKS`` (10) ticks earlier. It returns an immutable summary of
teams, scores, each team's score before this tick and 10 ticks before it (for
the Momentum switch), deaths (with a home-fall flag for deaths that scored
double), the champions that return this tick, and available run metadata. The
caller owns score history: the Replay Viewer reads its recorded frames and the
live service keeps a short tail in its session; the browser never keeps one.
Scores and outcomes stay owned by
recorded task evidence; which deaths scored double is owned by
``evaluation.events.team_deathmatch_home_fall_slots_v1``. Complete kill credit
uses the shared metric calculation on CPU; sparse evidence leaves credit
unknown. The module does not write files or advance the simulator.

Controller names come from the record (``evaluation.controller_names``) and are
shown exactly as recorded: each champion slot's name when the replay has them
(``slot_controller_names``; each team's ``display_name`` joins its slots'
names). Without them, Team B of a proven fixed scenario keeps its scenario
title, as before; else the recorded team name; else, for old records, today's
labels ("Recorded System" for a bare hash, a file-style ID cleaned up for
reading).

Agent POV note: every field here is global researcher data, attached to Agent
POV frames only for docked researcher panels. Nothing drawn on or over the
arena in Agent POV may read it. The scorebug beside the arena reads only its
public match facts (scores, threshold, tick), which are the same in every view.

Honors (the game-over Verdict card's leaders and Match Summary rows) are the
one exception built here for the arena: ``match_transition_totals_v1`` turns
one recorded transition into per-champion totals (Kill Contributions from the
summary's shared kill credit, Damage Done and Effective Healing Done with the
full metrics' formulas, Deaths, Red Zone Deaths and Ultimate Ability
Activations), ``MatchTotalsV1.plus`` adds them up, and ``match_honors_v1``
picks each honor's leaders. Both views see every champion, because Honors
come only once the game is over and nothing is left to hide then (a result
for one team is still possible). The live service and the Replay Viewer
service own the history and send the result as the presentation's
``match_honors`` field on the completion frame.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marl_battlegrounds.evaluation.controller_names import (
    recorded_slot_controller_names,
    recorded_team_controller_name,
)
from marl_battlegrounds.evaluation.events import team_deathmatch_home_fall_slots_v1
from marl_battlegrounds.evaluation.map_identity import RecordedMap, recorded_map
from marl_battlegrounds.evaluation.models import (
    AbilityActivatedEventV1,
    AgentDiedEventV1,
    AgentRespawnedEventV1,
    AssignedPolicySlot,
    AssignedPolicySlotV1,
    AssignedPolicySlotV2,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContext,
    EvaluationEpisodeContextV2,
    EvaluationFrame,
    LethalDamageContributionEventV1,
    RecipientHealthResolutionEventV1,
    SourceDamageOutputEventV1,
    SourceHealingOutputEventV1,
    TeamDeathmatchCompletedEventV1,
    TeamDeathmatchScoreChangedEventV1,
    red_zone_team_on_right,
)
from marl_battlegrounds.rendering.scene import (
    AbilityActivatedEventV2,
    AgentDiedEventV2,
    AgentRespawnedEventV2,
    LethalDamageContributionEventV2,
    RecipientHealthResolutionEventV2,
    SourceDamageOutputEventV2,
    SourceHealingOutputEventV2,
    TeamDeathmatchCompletedEventV2,
    TeamDeathmatchScoreChangedEventV2,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from marl_battlegrounds.evaluation.combat_metrics import CombatCredit

type MatchOutcome = Literal[
    "in_progress", "team_a_win", "team_b_win", "draw", "not_applicable"
]
type _Count = Annotated[int, Field(ge=0)]
type _Name = Annotated[str, Field(min_length=1)]

MOMENTUM_TICKS: Final = 10
"""How many ticks back the Momentum switch (hud_momentum) looks.

MatchSummaryV1.scores_ten_ticks_ago is each team's score this many frames
before the shown frame (or at frame zero in a younger game). Points scored
since then are "recent". One frame is one tick in a recorded or live game.
"""


class _MatchModel(BaseModel):
    """Strict immutable base that rejects fields outside the match-summary contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MatchTeamV1(_MatchModel):
    """One team's name, policy identities and recorded horizontal display side.

    display_name is the team's controller name as recorded, never rewritten:
    with per-slot names (see MatchSummaryV1.slot_controller_names) it joins
    the team's different slot names with " / " in slot order. Without them,
    Team B of a proven fixed scenario keeps its scenario controller title
    ("TDM Scenario 5 BETA Controller"), which wins over a recorded team name
    as it did before per-slot names; else the recorded team name; else the
    older labels (a built-in label, "Recorded System" for a bare hash, or a
    file-style ID cleaned up for reading). display_side is None for missing
    or non-left-right spawn banks. Such records keep the historical Team A
    left, Team B right layout.
    """

    team_id: Literal[1, 2]
    display_name: _Name
    display_side: Literal["left", "right"] | None = Field(default_factory=lambda: None)
    policy_ids: tuple[_Name, ...]
    checkpoint_digests: tuple[Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")], ...]


class MatchAgentV1(_MatchModel):
    """A public agent identity with its fixed recorded team and class."""

    public_agent_id: _Name
    team_id: Literal[1, 2]
    class_id: Literal[1, 2, 3, 4, 5]


class MatchDeathV1(MatchAgentV1):
    """An incoming death with complete opposing-team credit when evidence permits.

    Killing team and contributors are both None when attribution is unavailable;
    missing evidence must not be shown as a known empty contributor list.

    Attributes
    ----------
    killing_team_id : {1, 2} or None
        Team credited with the kill, or None when credit is unknown.
    contributors : tuple of MatchAgentV1 or None
        One to five opposing contributors, or None when credit is unknown.
    home_fall : bool
        True when this death started inside the victim's own team's Red Zone,
        so it gave the other team 2 points instead of 1. False for an ordinary
        fall, for a summary built without the transition start frame (the
        start position is then unknown), and for records written before this
        field existed (the default keeps them valid).
    """

    killing_team_id: Literal[1, 2] | None = Field(default_factory=lambda: None)
    contributors: tuple[MatchAgentV1, ...] | None = Field(
        default_factory=lambda: None, min_length=1, max_length=5
    )
    home_fall: bool = Field(default_factory=lambda: False)

    @model_validator(mode="after")
    def _validate_credit(self) -> Self:
        """Require complete credit, unique contributors, and an opposing killing
        team.
        """
        if (self.killing_team_id is None) != (self.contributors is None):
            raise ValueError(
                "death attribution must be complete or explicitly unavailable"
            )
        if self.contributors is not None:
            if self.killing_team_id == self.team_id or any(
                row.team_id != self.killing_team_id for row in self.contributors
            ):
                raise ValueError(
                    "kill contributors must belong to the opposing killing team"
                )
            if len({row.public_agent_id for row in self.contributors}) != len(
                self.contributors
            ):
                raise ValueError("kill contributors must be unique for each victim")
        return self


class MatchSummaryV1(_MatchModel):
    """Recorded match facts for one frame, with Team A before Team B.

    Scores and outcomes come from task evidence. Death rows describe only the
    incoming transition. Optional metadata remains absent when not recorded.

    Attributes
    ----------
    scores : tuple of (int, int)
        Team A and Team B scores after this tick.
    previous_scores : tuple of (int, int) or None
        Team A and Team B scores before this tick (at the transition start).
        Equal to ``scores`` on frame zero and on a tick where nobody scored.
        None only for payloads written before this field existed; browsers
        then treat the score as unchanged. Clients use it to hold the old score
        until the Fall beat.
    scores_ten_ticks_ago : tuple of (int, int) or None
        Team A and Team B scores ``MOMENTUM_TICKS`` (10) ticks before this
        one: at the frame 10 frames earlier, or at frame zero when the game is
        younger (so equal to ``scores`` on frame zero). A game that starts
        mid-way (frame zero at tick 91) counts from its own frame zero. The
        Momentum switch (hud_momentum) shows the points scored since then as
        recent. Never above ``previous_scores`` or ``scores``. None when the
        builder was not given the earlier scores (payloads written before this
        field existed); browsers then show no recent points. Scores are public
        match facts, the same in Agent POV and Oracle View.
    deaths : tuple of MatchDeathV1
        Champions that fell during the incoming transition, each once.
    returned_agent_ids : tuple of str
        Public IDs of the champions that return (respawn) during the incoming
        transition, in global slot order. Empty on frame zero, on ticks with
        no return, and for older payloads. Clients show a return at the
        Return beat.
    slot_controller_names : tuple of (str or None), or None
        The recorded controller of each champion slot, ten entries in global
        slot order (0 to 4 Team A, 5 to 9 Team B), exactly as recorded
        ("tdm-beta", "Manual"); None for an unused slot. The whole field is
        None when the replay records no slot names (replays saved before
        2 October 2026); clients then fall back to the team names. The Replay
        Viewer builds its controller lists from it. Who controls a champion is
        fixed before the game, so it reveals nothing in Agent POV.
    """

    schema_version: Literal[1]
    episode_id: _Name
    source_frame_index: _Count
    simulator_step_count: _Count
    task_mode: Literal[0, 1]
    score_threshold: _Count
    scores: tuple[_Count, _Count]
    outcome: MatchOutcome
    teams: tuple[MatchTeamV1, MatchTeamV1]
    deaths: tuple[MatchDeathV1, ...] = Field(default_factory=tuple, max_length=10)
    previous_scores: tuple[_Count, _Count] | None = Field(default_factory=lambda: None)
    scores_ten_ticks_ago: tuple[_Count, _Count] | None = Field(
        default_factory=lambda: None
    )
    returned_agent_ids: tuple[_Name, ...] = Field(default_factory=tuple, max_length=10)
    map: RecordedMap | None = Field(default_factory=lambda: None)
    observation_mode: Literal["shared_obs", "no_shared_obs"] | None = Field(
        default_factory=lambda: None
    )
    episode_limit: Annotated[int, Field(gt=0)] | None = Field(
        default_factory=lambda: None
    )
    root_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = Field(
        default_factory=lambda: None
    )
    episode_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = Field(
        default_factory=lambda: None
    )
    slot_controller_names: (
        Annotated[tuple[_Name | None, ...], Field(min_length=10, max_length=10)] | None
    ) = Field(default_factory=lambda: None)

    @model_validator(mode="after")
    def _validate_task(self) -> Self:
        """Keep team order, unique incoming deaths and returns, scores that never
        go down (10 ticks ago, then before this tick, then now), and
        task-specific score/result scope.
        """
        if tuple(team.team_id for team in self.teams) != (1, 2):
            raise ValueError("match teams must retain Team A then Team B order")
        if self.previous_scores is not None and any(
            before > after
            for before, after in zip(self.previous_scores, self.scores, strict=True)
        ):
            raise ValueError("a team's score cannot go down within one tick")
        if self.scores_ten_ticks_ago is not None and any(
            earlier > later
            for earlier, later in zip(
                self.scores_ten_ticks_ago,
                self.previous_scores or self.scores,
                strict=True,
            )
        ):
            raise ValueError("a team's score cannot go down over ten ticks")
        if len(set(self.returned_agent_ids)) != len(self.returned_agent_ids):
            raise ValueError("match returns must contain each returning agent once")
        sides = tuple(team.display_side for team in self.teams)
        if sides not in ((None, None), ("left", "right"), ("right", "left")):
            raise ValueError("match display sides must be opposite or both unavailable")
        if len({death.public_agent_id for death in self.deaths}) != len(self.deaths):
            raise ValueError("match deaths must contain each incoming recipient once")
        if self.task_mode == 0:
            if (
                self.score_threshold != 0
                or self.scores != (0, 0)
                or self.previous_scores not in (None, (0, 0))
                or self.scores_ten_ticks_ago not in (None, (0, 0))
                or any(death.home_fall for death in self.deaths)
            ):
                raise ValueError("combat diagnostics cannot claim TDM scores")
            if self.outcome != "not_applicable":
                raise ValueError("combat diagnostics have no TDM result")
        elif self.score_threshold == 0 or self.outcome == "not_applicable":
            raise ValueError("TDM requires its configured threshold and result scope")
        return self


_CONTROLLER_NAMES = {
    "manual": "Manual",
    "scripted": "Scripted scenario",
    "random_valid": "Random (legal actions)",
    "reactive_tdm": "Reactive TDM ALPHA",
    "scenario_5": "Reactive TDM BETA",
    "tdm_gamma": "Reactive TDM GAMMA",
}


@cache
def _combat_credit() -> Callable[..., CombatCredit]:
    """Load the shared calculation only when a death needs attribution."""
    import jax

    from marl_battlegrounds.evaluation.combat_metrics import combat_credit

    return cast("Callable[..., CombatCredit]", jax.jit(combat_credit))


def _death_contributors(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
    incoming_events: tuple[object, ...],
    dead_slots: set[int],
) -> dict[int, tuple[MatchAgentV1, ...]]:
    """Decode only observed credit inputs; sparse historical evidence stays unknown.

    Full global events already carry transition-start health and delivered effects.
    The tiny shared calculation runs on CPU, without reconstructing simulator
    state, computing full metrics or evaluating hypothetical rescue actions.
    """
    # Canonical global bundles start at zero and contain every event in order.
    # A sparse historical subset cannot establish complete support attribution.
    if tuple(getattr(event, "ordinal", None) for event in incoming_events) != tuple(
        range(len(incoming_events))
    ):
        return {}
    import numpy as np

    routing = np.zeros((10, 10), dtype=np.float32)
    healing = np.zeros_like(routing)
    contributed = np.zeros(10, dtype=np.bool_)
    health_before = np.zeros(10, dtype=np.float32)
    health_after = np.zeros(10, dtype=np.float32)
    total_healing = np.zeros(10, dtype=np.float32)
    total_damage = np.zeros(10, dtype=np.float32)
    attributed_damage = np.zeros(10, dtype=np.float32)
    healing_sources: set[int] = set()
    health_recipients: set[int] = set()
    for event in incoming_events:
        if not isinstance(
            event,
            (
                LethalDamageContributionEventV1,
                LethalDamageContributionEventV2,
                SourceHealingOutputEventV1,
                SourceHealingOutputEventV2,
                RecipientHealthResolutionEventV1,
                RecipientHealthResolutionEventV2,
            ),
        ):
            continue
        if (
            event.transition_id
            != f"{frame.episode_id}:transition:{frame.frame_index - 1}"
        ):
            raise ValueError(
                "kill credit must belong to the frame's incoming transition"
            )
        recipient = event.recipient_global_slot
        if recipient is None:
            raise ValueError("kill credit effects require their recorded recipient")
        if isinstance(
            event, (RecipientHealthResolutionEventV1, RecipientHealthResolutionEventV2)
        ):
            if recipient in health_recipients:
                raise ValueError(
                    "kill credit cannot repeat a recipient health resolution"
                )
            health_recipients.add(recipient)
            health_before[recipient] = event.transition_start_health
            health_after[recipient] = event.health_after_combat_resolution
            total_healing[recipient] = event.total_effective_healing
            total_damage[recipient] = event.total_effective_damage
            continue
        source = event.source_global_slot
        roster = context.roster[source]
        if not roster.configured_active:
            raise ValueError("kill credit effects require an active recorded source")
        routing[source, recipient] = 1
        if isinstance(event, (SourceHealingOutputEventV1, SourceHealingOutputEventV2)):
            if source in healing_sources:
                raise ValueError("kill credit cannot repeat a source healing output")
            healing_sources.add(source)
            healing[source, recipient] = np.float32(
                event.source_modified_healing_output
            ) * np.float32(event.recipient_healing_modifier)
        else:
            if contributed[source] or recipient not in dead_slots:
                raise ValueError(
                    "direct contribution must uniquely join an incoming death"
                )
            if (
                roster.configured_team_id
                == context.roster[recipient].configured_team_id
            ):
                raise ValueError("direct contribution must belong to the opposing team")
            if event.attributed_death_damage <= 0:
                raise ValueError(
                    "direct contribution requires positive recorded damage"
                )
            contributed[source] = True
            attributed_damage[recipient] += np.float32(event.attributed_death_damage)
    if not contributed.any():
        return {}
    # Missing source or recipient records can hide useful Priest support. Never
    # present an apparently complete contributor list from partial evidence.
    recorded_healing = healing.sum(axis=0)
    affected = set(np.flatnonzero(recorded_healing > 0)) | dead_slots
    dead = sorted(dead_slots)
    if (
        not affected.issubset(health_recipients)
        or not np.allclose(recorded_healing, total_healing, rtol=1e-6, atol=1e-5)
        # A removed final contribution can leave a contiguous ordinal prefix.
        # Every delivered damage amount on a lethal recipient earns direct credit.
        or not np.allclose(
            attributed_damage[dead], total_damage[dead], rtol=1e-6, atol=1e-5
        )
    ):
        return {}
    inputs = (
        np.asarray([row.class_id for row in context.roster], dtype=np.int32),
        health_before,
        healing,
        routing,
        total_healing,
        total_damage,
        health_after,
        contributed,
    )
    # Explicit CPU placement avoids touching the researcher's training GPU.
    import jax

    cpu = cast(Any, jax.devices("cpu")[0])
    credit = _combat_credit()(*jax.device_put(inputs, cpu))
    contributions = cast("NDArray[np.bool_]", np.asarray(credit.kill_contributions))
    result: dict[int, tuple[MatchAgentV1, ...]] = {}
    for recipient in sorted(dead_slots):
        rows = tuple(
            MatchAgentV1(
                public_agent_id=context.roster[source].public_agent_id,
                team_id=cast(
                    "Literal[1, 2]", context.roster[source].configured_team_id
                ),
                class_id=cast(
                    "Literal[1, 2, 3, 4, 5]", context.roster[source].class_id
                ),
            )
            for source in np.flatnonzero(contributions[:, recipient])
        )
        if rows:
            result[recipient] = rows
    return result


def _controller_name(
    context: EvaluationEpisodeContext,
    row: AssignedPolicySlot,
    team_id: int,
    system_name: str | None = None,
) -> str:
    """Name a recorded controller, keeping proven scenario names most specific.

    context and row hold recorded identity facts; team_id is the physical team.
    system_name is an optional saved team name. It is returned exactly as
    recorded ("tdm-beta" stays "tdm-beta") and takes precedence over older
    policy labels, but not a proven scenario controller. Exact policy IDs
    remain separate from this display text. Used only for records without
    per-slot controller names.
    """
    scenario = context.identity.scenario
    match = (
        None
        if scenario is None
        else re.fullmatch(
            r"(?:tdm-scenario-|custom-debugger-scenario:scenario_)([1-8])",
            scenario.identifier,
        )
    )
    if match is None and isinstance(context, EvaluationEpisodeContextV2):
        match = re.fullmatch(r"scenario_([1-8])", context.scenario_name or "")
    if match is None:
        names = tuple(
            row.value for row in context.aggregation_keys if row.name == "scenario"
        )
        match = re.fullmatch(r"scenario_([1-8])", names[0]) if len(names) == 1 else None
    variants = {
        "reactive-team-deathmatch-controller": "alpha",
        "scenario-5-pressure-controller": "beta",
    }
    variant = variants.get(row.algorithm_id or "") or variants.get(row.policy_id)
    if variant is None:
        key = f"team_{'a' if team_id == 1 else 'b'}_controller_identity"
        identities = tuple(
            binding.value for binding in context.aggregation_keys if binding.name == key
        )
        if len(identities) == 1:
            try:
                identity = ContentAddressedIdentityV1.model_validate_json(identities[0])
            except ValueError:
                pass
            else:
                variant = variants.get(identity.identifier)
    if team_id == 2 and match is not None and variant is not None:
        return f"TDM Scenario {match[1]} {variant.upper()} Controller"
    if system_name is not None:
        return system_name
    return _CONTROLLER_NAMES.get(row.policy_kind, _display_label(row.policy_id))


def _display_label(name: str) -> str:
    """Turn an older file-style policy ID into display text; keep hashes in provenance.

    name is a recorded policy ID, used only when the record has no recorded
    controller name. Unknown hash-only identities use a plain label; exact IDs
    remain in MatchTeamV1.policy_ids and its browser tooltip. Recorded names
    never pass through here: they are shown exactly as recorded.
    """
    if re.fullmatch(r"[0-9a-f]{32,64}(?::component-\d+)?", name):
        return "Recorded System"
    label = " ".join(name.replace("_", " ").replace("-", " ").split())
    return label[:1].upper() + label[1:]


@lru_cache(maxsize=128)
def _team_display_sides(
    width: float, pads: tuple[tuple[tuple[float, float], ...], ...] | None
) -> tuple[Literal["left", "right"] | None, Literal["left", "right"] | None]:
    """Read immutable spawn banks once; ambiguous layouts return two None values.

    width is the recorded map width. pads contains both five-pad banks, or None
    for older recordings. Only banks wholly on opposite sides of the middle are
    a horizontal layout. Red Zone need not be enabled. Actor positions are unused.
    """
    if pads is None:
        return None, None
    half = width / 2
    left = tuple(all(x < half for x, _ in bank) for bank in pads)
    right = tuple(all(x > half for x, _ in bank) for bank in pads)
    if not ((left[0] and right[1]) or (right[0] and left[1])):
        return None, None
    on_right = red_zone_team_on_right(width, [x for x, _ in pads[0]])
    return ("right", "left") if on_right else ("left", "right")


def build_match_summary_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
    incoming_events: tuple[object, ...] = (),
    *,
    transition_start_frame: EvaluationFrame | None = None,
    scores_ten_ticks_ago: tuple[int, ...] | None = None,
) -> MatchSummaryV1:
    """Package recorded match facts without adding new score or victory rules.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Recorded roster, task settings, identities, and static mechanics.
    frame : EvaluationFrame
        Current frame from the same episode.
    incoming_events : tuple of object, optional
        Recorded events entering the current frame, default empty. Accepts the
        evaluation events of a transition or the events of a canonical visual
        batch. Death credit is available only when the event evidence is
        complete enough to establish it.
    transition_start_frame : EvaluationFrame or None, optional, keyword-only
        The frame the incoming transition started from (``frame_index - 1``).
        It supplies ``previous_scores`` and the start positions that decide
        ``home_fall``. None (the default) derives ``previous_scores`` from the
        recorded score-change events (or keeps ``scores`` when there are none)
        and leaves every ``home_fall`` False because the start positions are
        unknown. Must be None on frame zero.
    scores_ten_ticks_ago : tuple of int or None, optional, keyword-only
        Two whole numbers: Team A and Team B scores at the frame
        ``MOMENTUM_TICKS`` (10) frames before ``frame``, or at frame zero when
        ``frame`` is younger; on frame zero itself, the frame's own scores.
        The caller owns this history: the Replay Viewer reads its recorded
        frames (``frames[max(0, index - MOMENTUM_TICKS)]``) and the live
        service its session's ``scores_ten_ticks_ago``. Copied into
        ``MatchSummaryV1.scores_ten_ticks_ago`` for the Momentum switch; a
        length other than two is refused there. None (the default) leaves that
        field None, so browsers show no recent points.

    Returns
    -------
    MatchSummaryV1
        Team A then Team B, current, previous and 10-ticks-ago scores,
        outcome, incoming deaths with home-fall flags, returning agents, and
        available map/seed/episode metadata. Missing attribution is None, not
        a claimed zero. ``episode_limit`` is the game's last tick in Team Deathmatch
        (``maximum_episode_steps``, also for a game that starts mid-way) and
        the expected number of transitions in combat diagnostics. Team names
        and ``slot_controller_names`` are the recorded controller names,
        exactly as recorded; records without per-slot names give
        ``slot_controller_names`` None and keep their team names or older
        labels.

    Raises
    ------
    ValueError
        If the frame and context disagree, the start frame is not the frame
        just before ``frame``, completion events conflict, a score-change event
        does not join the frame's score, incoming events refer to another
        transition or contain inconsistent attribution, or
        ``scores_ten_ticks_ago`` is not the frame's own scores on frame zero
        or is above the score before this tick (pydantic's ValidationError,
        a ValueError).

    Notes
    -----
    Reads recorded evidence. When deaths require complete credit calculation, the
    shared numerical helper runs on CPU. This does not compute a full metric table.
    The result is global researcher data: in Agent POV it may feed docked
    researcher panels only, never anything drawn on or over the arena.
    """
    if frame.episode_id != context.identity.episode_id:
        raise ValueError("match summary frame must join its episode context")
    if transition_start_frame is not None and (
        transition_start_frame.episode_id != frame.episode_id
        or transition_start_frame.frame_index != frame.frame_index - 1
    ):
        raise ValueError("match summary start frame must be the frame before it")
    if (
        frame.frame_index == 0
        and scores_ten_ticks_ago is not None
        and tuple(scores_ten_ticks_ago) != tuple(frame.snapshot.team_deathmatch_scores)
    ):
        raise ValueError("on frame zero the scores ten ticks ago are the scores")
    incoming_transition_id = f"{frame.episode_id}:transition:{frame.frame_index - 1}"
    task_mode = context.resolved_env_config.task_mode
    outcome: MatchOutcome = "in_progress" if task_mode == 1 else "not_applicable"
    completed = tuple(
        event
        for event in incoming_events
        if isinstance(
            event, (TeamDeathmatchCompletedEventV1, TeamDeathmatchCompletedEventV2)
        )
    )
    if len(completed) > 1:
        raise ValueError("match summary cannot have multiple completion events")
    if completed:
        if completed[0].transition_id != (
            f"{frame.episode_id}:transition:{frame.frame_index - 1}"
        ):
            raise ValueError(
                "match result must belong to the frame's incoming transition"
            )
        outcome = completed[0].outcome
    deaths: list[MatchDeathV1] = []
    death_events = tuple(
        event
        for event in incoming_events
        if isinstance(event, (AgentDiedEventV1, AgentDiedEventV2))
    )
    home_falls: frozenset[int] = frozenset()
    if transition_start_frame is not None and death_events:
        dead_slots = {event.recipient_global_slot for event in death_events}
        home_falls = team_deathmatch_home_fall_slots_v1(
            context,
            transition_start_frame.snapshot.agent_positions,
            tuple(slot in dead_slots for slot in range(len(context.roster))),
        )
    contributors = (
        _death_contributors(
            context,
            frame,
            incoming_events,
            {event.recipient_global_slot for event in death_events},
        )
        if death_events
        else {}
    )
    for event in death_events:
        if (
            event.transition_id
            != f"{frame.episode_id}:transition:{frame.frame_index - 1}"
        ):
            raise ValueError(
                "match death must belong to the frame's incoming transition"
            )
        roster = context.roster[event.recipient_global_slot]
        credit = contributors.get(event.recipient_global_slot)
        deaths.append(
            MatchDeathV1(
                public_agent_id=roster.public_agent_id,
                team_id=cast("Literal[1, 2]", roster.configured_team_id),
                class_id=cast("Literal[1, 2, 3, 4, 5]", roster.class_id),
                killing_team_id=None if credit is None else credit[0].team_id,
                contributors=credit,
                home_fall=event.recipient_global_slot in home_falls,
            )
        )
    returned_slots: set[int] = set()
    for event in incoming_events:
        if not isinstance(event, (AgentRespawnedEventV1, AgentRespawnedEventV2)):
            continue
        if event.transition_id != incoming_transition_id:
            raise ValueError(
                "match return must belong to the frame's incoming transition"
            )
        returned_slots.add(event.agent_global_slot)
    scores = cast("tuple[int, int]", frame.snapshot.team_deathmatch_scores)
    if transition_start_frame is not None:
        previous_scores = cast(
            "tuple[int, int]", transition_start_frame.snapshot.team_deathmatch_scores
        )
    else:
        previous = list(scores)
        for event in incoming_events:
            if not isinstance(
                event,
                (TeamDeathmatchScoreChangedEventV1, TeamDeathmatchScoreChangedEventV2),
            ):
                continue
            if (
                event.transition_id != incoming_transition_id
                or event.successor_score != scores[event.team_index]
            ):
                raise ValueError("match score change must join the frame's score")
            previous[event.team_index] = event.previous_score
        previous_scores = (previous[0], previous[1])
    display_sides = _team_display_sides(
        context.resolved_env_config.map_width,
        context.resolved_env_config.team_spawn_pad_positions,
    )
    slot_names = recorded_slot_controller_names(context)
    teams: list[MatchTeamV1] = []
    for team_id in (1, 2):
        assignments = tuple(
            assignment
            for roster, assignment in zip(
                context.roster, context.policy_assignments, strict=True
            )
            if roster.configured_team_id == team_id
            and isinstance(assignment, (AssignedPolicySlotV1, AssignedPolicySlotV2))
        )
        policy_ids = tuple(dict.fromkeys(row.policy_id for row in assignments))
        system_name = recorded_team_controller_name(
            context, f"team_{'a' if team_id == 1 else 'b'}"
        )
        if slot_names is not None:
            # New records name every champion slot; show them as recorded.
            start = 0 if team_id == 1 else 5
            names = tuple(
                dict.fromkeys(
                    name for name in slot_names[start : start + 5] if name is not None
                )
            )
        else:
            names = tuple(
                dict.fromkeys(
                    _controller_name(context, row, team_id, system_name)
                    for row in assignments
                )
            )
        if not names and system_name is not None:
            names = (system_name,)
        teams.append(
            MatchTeamV1(
                team_id=team_id,
                display_side=display_sides[team_id - 1],
                display_name=" / ".join(names) if names else "No active policy",
                policy_ids=policy_ids,
                checkpoint_digests=tuple(
                    dict.fromkeys(
                        row.checkpoint_digest
                        for row in assignments
                        if row.checkpoint_digest is not None
                    )
                ),
            )
        )
    return MatchSummaryV1(
        schema_version=1,
        episode_id=frame.episode_id,
        source_frame_index=frame.frame_index,
        simulator_step_count=frame.simulator_step_count,
        task_mode=task_mode,
        score_threshold=context.resolved_env_config.team_deathmatch_score_threshold,
        scores=scores,
        previous_scores=previous_scores,
        scores_ten_ticks_ago=(
            None
            if scores_ten_ticks_ago is None
            else cast(
                "tuple[int, int]", tuple(int(score) for score in scores_ten_ticks_ago)
            )
        ),
        outcome=outcome,
        teams=(teams[0], teams[1]),
        deaths=tuple(deaths),
        returned_agent_ids=tuple(
            context.roster[slot].public_agent_id for slot in sorted(returned_slots)
        ),
        map=recorded_map(context),
        observation_mode=context.execution_information_mode,
        # The tick at which the game ends: in Team Deathmatch the configured
        # last tick (validation proves initial tick + expected_horizon equals
        # it, so a game started mid-way, such as at tick 91, still reads "of
        # 300"); otherwise the expected number of transitions.
        episode_limit=(
            context.resolved_env_config.maximum_episode_steps
            if task_mode == 1
            else context.expected_horizon
        ),
        root_seed=context.seed_protocol.root_seed,
        episode_seed=context.seed_protocol.episode_seed,
        slot_controller_names=slot_names,
    )


type HonorMetric = Literal[
    "kill_contributions", "damage_done", "effective_healing_done"
]
type SummaryMetric = Literal[
    "kill_contributions",
    "deaths",
    "damage_done",
    "effective_healing_done",
    "red_zone_deaths",
    "ultimate_activations",
]

HONOR_METRICS: Final[tuple[HonorMetric, ...]] = (
    "kill_contributions",
    "damage_done",
    "effective_healing_done",
)
"""The metrics the Verdict card's Honors name, in card order (the store's three)."""

SUMMARY_METRICS: Final[tuple[SummaryMetric, ...]] = (
    "kill_contributions",
    "deaths",
    "damage_done",
    "effective_healing_done",
    "red_zone_deaths",
    "ultimate_activations",
)
"""The per-champion columns of the Match Summary rows, in column order."""

_SLOTS: Final = 10
"""Global champion slots: 0 to 4 Team A, 5 to 9 Team B."""


@dataclass(frozen=True, slots=True)
class MatchTotalsV1:
    """Per-champion totals over the transitions added so far, by global slot.

    Every field holds ten values in global slot order (0 to 4 Team A, 5 to 9
    Team B); an inactive slot keeps zeros. The meanings follow the metric
    catalog's agent metrics of the same names: ``kill_contributions`` counts
    falls the champion helped with (damage on the tick of the fall, or a
    Priest's useful heal on one of those attackers, the shared kill credit);
    ``deaths`` counts its falls; ``damage_done`` is the damage it delivered
    in hit points, including damage beyond a target's remaining life;
    ``effective_healing_done`` is its Priest healing in hit points minus its
    share of the excess; ``red_zone_deaths`` counts its falls inside its own
    team's Red Zone; ``ultimate_activations`` counts its accepted Ultimates.

    Attributes
    ----------
    kill_contributions : tuple of int
        Default ten zeros.
    deaths : tuple of int
        Default ten zeros.
    damage_done : tuple of float
        Default ten zeros.
    effective_healing_done : tuple of float
        Default ten zeros.
    red_zone_deaths : tuple of int
        Default ten zeros.
    ultimate_activations : tuple of int
        Default ten zeros.
    kill_credit_known : bool
        False once any added fall had unknown kill credit (sparse evidence);
        Kill Contributions is then not shown. Default True.
    """

    kill_contributions: tuple[int, ...] = (0,) * _SLOTS
    deaths: tuple[int, ...] = (0,) * _SLOTS
    damage_done: tuple[float, ...] = (0.0,) * _SLOTS
    effective_healing_done: tuple[float, ...] = (0.0,) * _SLOTS
    red_zone_deaths: tuple[int, ...] = (0,) * _SLOTS
    ultimate_activations: tuple[int, ...] = (0,) * _SLOTS
    kill_credit_known: bool = True

    def plus(self, other: MatchTotalsV1) -> MatchTotalsV1:
        """Return these totals with another record added slot by slot.

        Parameters
        ----------
        other : MatchTotalsV1
            Totals to add (one transition's, or a run of them).

        Returns
        -------
        MatchTotalsV1
            A new record; ``kill_credit_known`` is True only when both are.
            Neither input changes.
        """
        return MatchTotalsV1(
            kill_contributions=_added(
                self.kill_contributions, other.kill_contributions
            ),
            deaths=_added(self.deaths, other.deaths),
            damage_done=_added(self.damage_done, other.damage_done),
            effective_healing_done=_added(
                self.effective_healing_done, other.effective_healing_done
            ),
            red_zone_deaths=_added(self.red_zone_deaths, other.red_zone_deaths),
            ultimate_activations=_added(
                self.ultimate_activations, other.ultimate_activations
            ),
            kill_credit_known=self.kill_credit_known and other.kill_credit_known,
        )


def _added[T: (int, float)](left: tuple[T, ...], right: tuple[T, ...]) -> tuple[T, ...]:
    """Return two ten-slot tuples added slot by slot."""
    return tuple(a + b for a, b in zip(left, right, strict=True))


def match_transition_totals_v1(
    context: EvaluationEpisodeContext,
    summary: MatchSummaryV1 | None,
    incoming_events: Iterable[object],
) -> MatchTotalsV1:
    """Count one recorded transition's per-champion totals for Honors.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        The recorded episode context; its roster maps public IDs to slots.
    summary : MatchSummaryV1 or None
        The match summary of the transition's successor frame, built with its
        start frame (``build_match_summary_v1(..., transition_start_frame=)``)
        so its deaths carry kill credit and home-fall flags. Its deaths give
        Deaths, Red Zone Deaths and Kill Contributions. None is allowed only
        for a transition without a fall (it saves building a summary).
    incoming_events : iterable of object
        The same transition's recorded events (evaluation events or a
        canonical visual batch's). Damage Done reads ``source_damage_output``
        (modified output times the recipient's modifier, as full metrics
        route it); Effective Healing Done reads ``source_healing_output`` and
        each recipient's ``recipient_health_resolution`` (healing minus the
        source's share of the excess, the excess split in proportion to the
        healing delivered); Ultimate Ability Activations reads
        ``ability_activated`` with component "ultimate". Other events are
        ignored.

    Returns
    -------
    MatchTotalsV1
        This transition's totals. ``kill_credit_known`` is False when a fall's
        credit is unknown in the summary.

    Raises
    ------
    ValueError
        ``summary`` is None but the events hold a fall.

    Notes
    -----
    Pure Python over at most a few dozen events: no JAX, file or simulator
    work (the summary already ran the shared kill-credit calculation).
    """
    events = tuple(incoming_events)
    if summary is None and any(
        isinstance(event, (AgentDiedEventV1, AgentDiedEventV2)) for event in events
    ):
        raise ValueError("a transition with a fall needs its match summary")
    slot_of = {row.public_agent_id: row.global_slot for row in context.roster}
    kills = [0] * _SLOTS
    deaths = [0] * _SLOTS
    red_zone = [0] * _SLOTS
    known = True
    for death in () if summary is None else summary.deaths:
        fallen = slot_of[death.public_agent_id]
        deaths[fallen] += 1
        red_zone[fallen] += 1 if death.home_fall else 0
        if death.contributors is None:
            known = False
            continue
        for helper in death.contributors:
            kills[slot_of[helper.public_agent_id]] += 1
    damage = [0.0] * _SLOTS
    ultimates = [0] * _SLOTS
    delivered: list[tuple[int, int, float]] = []
    resolution: dict[int, tuple[float, float, float, float]] = {}
    for event in events:
        if isinstance(event, (SourceDamageOutputEventV1, SourceDamageOutputEventV2)):
            if event.recipient_global_slot is not None:
                damage[event.source_global_slot] += float(
                    event.source_modified_damage_output
                ) * float(event.recipient_damage_modifier)
        elif isinstance(
            event, (SourceHealingOutputEventV1, SourceHealingOutputEventV2)
        ):
            if event.recipient_global_slot is not None:
                delivered.append(
                    (
                        event.source_global_slot,
                        event.recipient_global_slot,
                        float(event.source_modified_healing_output)
                        * float(event.recipient_healing_modifier),
                    )
                )
        elif isinstance(
            event,
            (RecipientHealthResolutionEventV1, RecipientHealthResolutionEventV2),
        ):
            resolution[event.recipient_global_slot] = (
                float(event.transition_start_health),
                float(event.total_effective_healing),
                float(event.total_effective_damage),
                float(event.health_after_combat_resolution),
            )
        elif (
            isinstance(event, (AbilityActivatedEventV1, AbilityActivatedEventV2))
            and event.ability_component == "ultimate"
        ):
            ultimates[event.source_global_slot] += 1
    healing = [0.0] * _SLOTS
    for source, recipient, amount in delivered:
        start, total_healing, total_damage, after = resolution.get(
            recipient, (0.0, 0.0, 0.0, 0.0)
        )
        excess = min(
            max(start + (total_healing - total_damage) - after, 0.0), total_healing
        )
        useful = 1.0 - (excess / total_healing if total_healing > 0 else 0.0)
        healing[source] += amount * useful
    return MatchTotalsV1(
        kill_contributions=tuple(kills),
        deaths=tuple(deaths),
        damage_done=tuple(damage),
        effective_healing_done=tuple(healing),
        red_zone_deaths=tuple(red_zone),
        ultimate_activations=tuple(ultimates),
        kill_credit_known=known,
    )


class MatchMetricLabelV1(_MatchModel):
    """One Match Summary column's metric key and its catalog label.

    Attributes
    ----------
    metric : SummaryMetric
        The agent metric's stem, one of ``SUMMARY_METRICS``.
    label : str
        Its Title Case label from the metric catalog (``Damage Done``).
    """

    metric: SummaryMetric
    label: _Name


class MatchChampionTotalsV1(MatchAgentV1):
    """One champion's game totals for the Match Summary rows.

    Values are whole numbers (damage and healing rounded half up, in hit
    points). ``kill_contributions`` is None when some fall's kill credit was
    unknown. See ``MatchTotalsV1`` for each meaning.
    """

    kill_contributions: _Count | None
    deaths: _Count
    damage_done: _Count
    effective_healing_done: _Count
    red_zone_deaths: _Count
    ultimate_activations: _Count


class MatchHonorV1(_MatchModel):
    """One Honor of the Verdict card: a metric and its leader or tied leaders.

    Attributes
    ----------
    metric : HonorMetric
        One of ``HONOR_METRICS``.
    label : str
        The metric's catalog label (``Kill Contributions``).
    value : int
        The leaders' shared whole-number value, 1 or more.
    leaders : tuple of MatchAgentV1
        Every champion of this view that has the value, in global slot order
        (one to ten).
    """

    metric: HonorMetric
    label: _Name
    value: Annotated[int, Field(gt=0)]
    leaders: tuple[MatchAgentV1, ...] = Field(min_length=1, max_length=10)


class MatchHonorsV1(_MatchModel):
    """The game-over Honors and Match Summary rows of one view.

    Attributes
    ----------
    schema_version : Literal[1]
        Version 1.
    team_id : {1, 2} or None
        None for every champion (both views at game over); 1 or 2 when only
        that team's own champions are counted, named and listed.
    labels : tuple of MatchMetricLabelV1
        One label per ``SUMMARY_METRICS`` key, in that order.
    honors : tuple of MatchHonorV1
        At most three, in ``HONOR_METRICS`` order; an honor nobody earned
        (best value 0) or whose credit is unknown is left out.
    champions : tuple of MatchChampionTotalsV1
        The counted champions' totals, in global slot order.

    Raises
    ------
    ValueError
        A champion is listed twice or is outside the view's team, a leader is
        not a listed champion, an honor's value is not the best value of its
        metric among the champions, or the labels are not the six columns in
        order (pydantic's ValidationError).
    """

    schema_version: Literal[1]
    team_id: Literal[1, 2] | None
    labels: tuple[MatchMetricLabelV1, ...]
    honors: tuple[MatchHonorV1, ...] = Field(max_length=3)
    champions: tuple[MatchChampionTotalsV1, ...] = Field(max_length=10)

    @model_validator(mode="after")
    def _validate_honors(self) -> Self:
        """Keep the view's team, the leaders and the best values consistent."""
        if tuple(row.metric for row in self.labels) != SUMMARY_METRICS:
            raise ValueError("honors labels must name the six summary columns in order")
        ids = [row.public_agent_id for row in self.champions]
        if len(set(ids)) != len(ids):
            raise ValueError("honors champions must be listed once each")
        if self.team_id is not None and any(
            row.team_id != self.team_id for row in self.champions
        ):
            raise ValueError("one team's honors count only that team's champions")
        by_id = {row.public_agent_id: row for row in self.champions}
        for honor in self.honors:
            values = [getattr(row, honor.metric) for row in self.champions]
            best = max((value for value in values if value is not None), default=0)
            if honor.value != best or any(
                leader.public_agent_id not in by_id
                or getattr(by_id[leader.public_agent_id], honor.metric) != best
                for leader in honor.leaders
            ):
                raise ValueError("an honor must name the champions with its best value")
        return self


@cache
def _summary_labels() -> tuple[MatchMetricLabelV1, ...]:
    """Read each summary column's label from the metric catalog (once)."""
    from marl_battlegrounds.evaluation.metric_catalog import METRIC_COLUMNS_BY_NAME

    return tuple(
        MatchMetricLabelV1(
            metric=metric, label=METRIC_COLUMNS_BY_NAME[f"agent_0_{metric}"].label
        )
        for metric in SUMMARY_METRICS
    )


def _whole(value: float) -> int:
    """Round a nonnegative amount half up to a whole number."""
    return math.floor(value + 0.5)


def match_honors_v1(
    context: EvaluationEpisodeContext,
    totals: MatchTotalsV1,
    *,
    team_id: Literal[1, 2] | None,
) -> MatchHonorsV1:
    """Pick one view's Honors and Match Summary rows from the game's totals.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        The recorded episode context: its configured-active roster rows are
        the champions counted.
    totals : MatchTotalsV1
        The whole game's totals (every transition's, added with ``plus``).
    team_id : {1, 2} or None, keyword-only
        None for every active champion, which both servers use for both views
        at game over (nothing is left to hide then). 1 or 2: only that team's
        own champions are listed and can lead.

    Returns
    -------
    MatchHonorsV1
        Damage and healing rounded half up to whole hit points. Each honor of
        ``HONOR_METRICS`` names every champion with the best rounded value;
        an honor whose best value is 0, or Kill Contributions when some kill
        credit was unknown, is left out. Labels come from the metric catalog.

    Notes
    -----
    Pure; reads the totals and roster only. A result for one team holds only
    that team's own champions, whose totals count their own blows, heals,
    falls and Ultimates.
    """
    champions: list[MatchChampionTotalsV1] = []
    for row in context.roster:
        if not row.configured_active:
            continue
        if team_id is not None and row.configured_team_id != team_id:
            continue
        slot = row.global_slot
        champions.append(
            MatchChampionTotalsV1(
                public_agent_id=row.public_agent_id,
                team_id=cast("Literal[1, 2]", row.configured_team_id),
                class_id=cast("Literal[1, 2, 3, 4, 5]", row.class_id),
                kill_contributions=(
                    totals.kill_contributions[slot]
                    if totals.kill_credit_known
                    else None
                ),
                deaths=totals.deaths[slot],
                damage_done=_whole(totals.damage_done[slot]),
                effective_healing_done=_whole(totals.effective_healing_done[slot]),
                red_zone_deaths=totals.red_zone_deaths[slot],
                ultimate_activations=totals.ultimate_activations[slot],
            )
        )
    labels = _summary_labels()
    label_of = {row.metric: row.label for row in labels}
    honors: list[MatchHonorV1] = []
    for metric in HONOR_METRICS:
        values = [getattr(row, metric) for row in champions]
        known = [value for value in values if value is not None]
        best = max(known, default=0)
        if best <= 0 or len(known) != len(values):
            continue
        honors.append(
            MatchHonorV1(
                metric=metric,
                label=label_of[metric],
                value=best,
                leaders=tuple(
                    MatchAgentV1(
                        public_agent_id=row.public_agent_id,
                        team_id=row.team_id,
                        class_id=row.class_id,
                    )
                    for row in champions
                    if getattr(row, metric) == best
                ),
            )
        )
    return MatchHonorsV1(
        schema_version=1,
        team_id=team_id,
        labels=labels,
        honors=tuple(honors),
        champions=tuple(champions),
    )
