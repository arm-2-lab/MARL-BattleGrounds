"""Count the history facts that the Herald Cards need, one view at a time.

The Herald Cards (the ``match_moments`` switch, Moments group) name the big
moments of a match above the arena. The browser works out most moments from
the one frame it holds (``web/visual_debugger/src/looks/heralds.js``,
``heraldMoments``). A few moments need facts that one frame cannot give: the
match's history, or blows the browser frame does not carry. This module is
the one owner of those facts. The browser draws the presentation field
``match_moments`` (``MatchMomentFactsV1``) it receives and never counts
history itself.

The facts, for the shown frame and its view:

- ``fresh_start``: the match began at tick 0 with the score 0-0. First Blood
  is told only in such a match (a scenario or custom start skips it).
- ``largest_lead_before``: the largest lead either team held at the start of
  this tick or at the start of any earlier tick. The browser names a
  Comeback ("Scores Level") when the scores become level on this tick after
  a lead of 3 or more.
- ``runs``: a team's 5th, 10th, 15th or 20th (every 5th) takedown in a row
  with no loss of its own, reached on this tick ("5 Unanswered"). Any loss, a
  trade included, starts the team's count again.
- ``rescues``: champions that a Priest's heal kept alive through blows that
  would otherwise have killed them on this tick ("Rescue").

Who calls it:

- The Replay Viewer service (``viewer.replay_service``) keeps one table per
  view, grown frame by frame, so seeks, view switches and page reloads read
  the same exact value.
- The live BattleClient and DevClient service keeps every view's history on
  its session (``_battle_client.model.DebuggerSession.match_moments``),
  advanced once per transition by
  ``_battle_client.live_presentation.advance_live_match_history_v1``.

The rules, per view (one "audience"):

- Oracle View reads every recorded fact.
- Agent POV reads only what the viewing champion's team saw. Runs count only
  the falls the team saw (the same falls its Fallen feed names). A rescue is
  told only when the team saw the rescued champion at both ends of the tick,
  saw a Priest's heal land on it, and saw every blow that hit it this tick
  (each damage source was seen at the start of the tick and its damaging
  Basic or Ultimate on that champion is in the team's own visual events). So
  an Agent POV rescue never reveals a blow from an unseen champion.
- The start facts (``fresh_start`` and ``largest_lead_before``) read only the
  public score and tick, which every view's scoreboard shows.

Everything is plain Python on small integers: no JAX, file, network or
simulator work.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marl_battlegrounds.evaluation._replays.wire_shapes import MAX_AGENT_SLOTS_V1
from marl_battlegrounds.evaluation.models import (
    AbilityActivatedEventV1,
    RecipientHealthResolutionEventV1,
    SourceDamageOutputEventV1,
)
from marl_battlegrounds.rendering.scene import (
    AbilityActivatedEventV2,
    RecipientHealthResolutionEventV2,
    SourceDamageOutputEventV2,
)
from marl_battlegrounds.rendering.vocabulary import PRIEST_CLASS_ID
from marl_battlegrounds.viewer.champion_streaks import DAMAGING_ULTIMATE_CLASS_IDS

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.models import EvaluationEpisodeContext
    from marl_battlegrounds.rendering.authorized_presentation import (
        AgentPovVisualIncomingSummaryV1,
    )

SLOT_COUNT: Final = MAX_AGENT_SLOTS_V1
"""Global champion slots (10: 0 to 4 Team A, 5 to 9 Team B)."""

RUN_EVERY: Final = 5
"""A team's run of unanswered takedowns is announced at every multiple of this."""

NO_RUNS: Final[tuple[int, int]] = (0, 0)
"""Run counts before any takedown: (Team A, Team B)."""

type RescueFact = tuple[int, bool]
"""One rescued champion on one tick: (global slot 0-9, by_ultimate)."""


class TeamRunV1(BaseModel):
    """One team that reached a run of unanswered takedowns on this tick.

    Attributes
    ----------
    team_id : {1, 2}
        The team that made the run (1 Team A, 2 Team B).
    takedowns : int
        The run reached on this tick: the team has now taken down this many
        enemy champions in a row without losing one of its own (in this
        view). A multiple of ``RUN_EVERY`` (5, 10, 15, 20 and so on).

    Raises
    ------
    pydantic.ValidationError
        ``takedowns`` is below 5 or not a multiple of 5, or ``team_id`` is
        not 1 or 2.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    team_id: Literal[1, 2]
    takedowns: Annotated[int, Field(ge=RUN_EVERY)]

    @field_validator("takedowns")
    @classmethod
    def _every_fifth(cls, takedowns: int) -> int:
        """Accept only a multiple of ``RUN_EVERY``; return it unchanged."""
        if takedowns % RUN_EVERY:
            raise ValueError(f"a run must be a multiple of {RUN_EVERY}")
        return takedowns


class RescueV1(BaseModel):
    """One champion a Priest's heal kept alive through lethal blows on this tick.

    Attributes
    ----------
    public_agent_id : str
        The rescued champion's public ID, as in the presentation's scene and
        events.
    team_id : {1, 2}
        The rescued champion's team.
    class_id : {1, 2, 3, 4, 5}
        The rescued champion's class (Mage, Warrior, Hunter, Rogue, Priest).
    by_ultimate : bool
        True when a Priest's Ultimate (Holy Word: Salvation) healed it on
        this tick (in Agent POV: a Salvation the team saw); False when only
        Lesser Heal did. The card's eyebrow names that heal.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    public_agent_id: Annotated[str, Field(min_length=1)]
    team_id: Literal[1, 2]
    class_id: Literal[1, 2, 3, 4, 5]
    by_ultimate: bool


class MatchMomentFactsV1(BaseModel):
    """The Herald Cards' history facts for one frame and its view.

    This is the presentation field ``match_moments``. The browser's moment
    finder reads it with the frame's own facts; see the module description.

    Attributes
    ----------
    fresh_start : bool
        True when the match's first frame was tick 0 with the score 0-0.
        False on frame zero itself (no tick has been played yet, so no
        moment can be told) and for every other start.
    largest_lead_before : int
        The largest lead (absolute score difference, 0 or more) either team
        held at the start of this tick or at the start of any earlier tick.
        0 on frame zero.
    runs : tuple of TeamRunV1
        Teams that reached a run of 5, 10, 15 or 20 (every 5th) unanswered
        takedowns on this tick, in this view; at most one entry per team.
        Empty on most ticks.
    rescues : tuple of RescueV1
        Champions a Priest's heal rescued on this tick, in this view, by
        global slot; each at most once. Empty on most ticks.

    Raises
    ------
    ValueError
        A team appears twice in ``runs`` or a champion twice in ``rescues``
        (pydantic's ValidationError), or ``largest_lead_before`` is negative.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    fresh_start: bool
    largest_lead_before: Annotated[int, Field(ge=0)]
    runs: tuple[TeamRunV1, ...]
    rescues: tuple[RescueV1, ...]

    @model_validator(mode="after")
    def _validate_unique(self) -> Self:
        """Require each team at most once in runs and each champion once in rescues."""
        teams = [row.team_id for row in self.runs]
        names = [row.public_agent_id for row in self.rescues]
        if len(set(teams)) != len(teams):
            raise ValueError("match moment runs must name each team at most once")
        if len(set(names)) != len(names):
            raise ValueError("match moment rescues must name each champion once")
        return self


def oracle_tick_rescues_v1(
    context: EvaluationEpisodeContext, events: Iterable[object]
) -> tuple[RescueFact, ...]:
    """Read Oracle View's rescues from one transition's recorded events.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        The recorded episode context; its roster gives each slot's class.
    events : iterable of object
        The recorded events of one transition: evaluation events (live
        ``EvaluationTransitionV1.events`` or a replay's) or the events of a
        canonical visual batch. Only recipient_health_resolution and
        ability_activated events (V1 or V2) are read; others are ignored.

    Returns
    -------
    tuple of (int, bool)
        (global slot, by_ultimate) per rescued champion, by slot. A champion
        is rescued when its health resolution shows that it was alive at the
        start of the tick (start health above 0), that the tick's damage
        alone would have killed it (start health minus total effective
        damage is 0 or less), that it was healed (total effective healing
        above 0) and that it lived (health after combat above 0), and a
        Priest's Basic or Ultimate named it as recipient. by_ultimate is
        True when one of those heals was a Priest's Ultimate (Holy Word:
        Salvation). This is the same rule as the Saved Beat. Empty on most
        ticks.

    Notes
    -----
    Pure; reads the events only.
    """
    rows = tuple(events)
    rescued: list[int] = []
    for event in rows:
        if not isinstance(
            event, (RecipientHealthResolutionEventV1, RecipientHealthResolutionEventV2)
        ):
            continue
        start = float(event.transition_start_health)
        if (
            start > 0
            and float(event.total_effective_healing) > 0
            and start - float(event.total_effective_damage) <= 0
            and float(event.health_after_combat_resolution) > 0
        ):
            rescued.append(event.recipient_global_slot)
    facts: list[RescueFact] = []
    for slot in sorted(set(rescued)):
        heals = [
            event.ability_component
            for event in rows
            if isinstance(event, (AbilityActivatedEventV1, AbilityActivatedEventV2))
            and event.recipient_global_slot == slot
            and context.roster[event.source_global_slot].class_id == PRIEST_CLASS_ID
        ]
        if heals:
            facts.append((slot, "ultimate" in heals))
    return tuple(facts)


def agent_pov_tick_rescues_v1(
    context: EvaluationEpisodeContext,
    oracle_rescues: tuple[RescueFact, ...],
    events: Iterable[object],
    visual_events: AgentPovVisualIncomingSummaryV1,
) -> tuple[RescueFact, ...]:
    """Keep only the rescues one Agent POV recipient's team saw whole.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        The recorded episode context; its roster maps public IDs to global
        slots.
    oracle_rescues : tuple of (int, bool)
        The tick's rescues from ``oracle_tick_rescues_v1``.
    events : iterable of object
        The same transition's recorded events. Only source_damage_output
        events (V1 or V2) are read, to list every source that damaged a
        rescued champion.
    visual_events : AgentPovVisualIncomingSummaryV1
        That recipient's own Agent POV visual events for the transition, as
        ``rendering.authorized_presentation.
        build_agent_pov_visual_incoming_summary_v1`` builds them (the same
        filter the arena uses).

    Returns
    -------
    tuple of (int, bool)
        The rescues this team saw, by slot. A rescue is kept only when the
        team saw the rescued champion at the start and at the end of the
        tick (its trajectory has both anchors), saw a Priest's Basic or
        Ultimate on it (an ability_activated row in its own visual events),
        and saw every blow that hit it: each source with recorded damage to
        it above 0 has a damaging Basic or Ultimate on it in the team's own
        visual events (a Basic is damaging unless its source is a Priest; an
        Ultimate is damaging for ``DAMAGING_ULTIMATE_CLASS_IDS``).
        by_ultimate is True when a Salvation the team saw healed it.

    Notes
    -----
    Pure. A rescue with a blow from a champion the team did not see is
    dropped, so the card never reveals that blow. Empty when
    ``oracle_rescues`` is empty.
    """
    if not oracle_rescues:
        return ()
    slot_by_public_id = {row.public_agent_id: row.global_slot for row in context.roster}
    trajectories = {
        slot_by_public_id[row.agent_public_agent_id]: row
        for row in visual_events.agent_phase_trajectories
    }
    class_by_slot = {slot: row.agent_class_id for slot, row in trajectories.items()}
    damaged_by: dict[int, set[int]] = {}
    for event in events:
        if (
            isinstance(event, (SourceDamageOutputEventV1, SourceDamageOutputEventV2))
            and event.recipient_global_slot is not None
            and float(event.source_modified_damage_output)
            * float(event.recipient_damage_modifier)
            > 0
        ):
            damaged_by.setdefault(event.recipient_global_slot, set()).add(
                event.source_global_slot
            )
    seen_heals: dict[int, list[str]] = {}
    seen_blows: dict[int, set[int]] = {}
    for event in visual_events.events:
        if event.event_kind != "ability_activated" or event.recipient_anchor is None:
            continue
        source = slot_by_public_id[event.source_anchor.public_agent_id]
        recipient = slot_by_public_id[event.recipient_anchor.public_agent_id]
        class_id = class_by_slot.get(source)
        if class_id is None:
            continue
        if class_id == PRIEST_CLASS_ID:
            seen_heals.setdefault(recipient, []).append(event.ability_component)
            continue
        damaging = (
            class_id in DAMAGING_ULTIMATE_CLASS_IDS
            if event.ability_component == "ultimate"
            else True
        )
        if damaging:
            seen_blows.setdefault(recipient, set()).add(source)
    kept: list[RescueFact] = []
    for slot, _by_ultimate in oracle_rescues:
        trajectory = trajectories.get(slot)
        heals = seen_heals.get(slot, [])
        if (
            trajectory is None
            or trajectory.transition_start is None
            or trajectory.successor is None
            or not heals
            or not damaged_by.get(slot, set()) <= seen_blows.get(slot, set())
        ):
            continue
        kept.append((slot, "ultimate" in heals))
    return tuple(kept)


def advance_team_runs_v1(
    runs: tuple[int, int],
    context: EvaluationEpisodeContext,
    falls: frozenset[int],
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    """Add one tick's falls to one view's runs of unanswered takedowns.

    Parameters
    ----------
    runs : tuple of (int, int)
        (Team A, Team B): each team's takedowns in a row since it last lost
        a champion (in this view), each 0 or more. Use ``NO_RUNS`` on frame
        zero.
    context : EvaluationEpisodeContext
        The recorded episode context; its roster gives each slot's team.
    falls : frozenset of int
        Global slots (0-9) of the champions this view saw fall on the tick
        (``TickTakedownsV1.falls`` of viewer.champion_streaks).

    Returns
    -------
    runs : tuple of (int, int)
        The runs after the tick. A team that lost a champion goes back to 0
        (a trade resets both); otherwise it gains one per enemy fall.
    reached : tuple of (int, int)
        (team ID, run) for every team whose run first passed a multiple of
        ``RUN_EVERY`` on this tick, Team A first, with the largest multiple
        passed (a tick adds at most 5, one per enemy, so it passes at most
        one). Empty on most ticks.

    Raises
    ------
    ValueError
        ``runs`` is not two whole numbers 0 or more, or a fall is not a
        global slot 0-9.
    """
    if (
        type(runs) is not tuple
        or len(runs) != 2
        or any(type(value) is not int or value < 0 for value in runs)
    ):
        raise ValueError("team runs must be two whole numbers 0 or more")
    if any(type(slot) is not int or not 0 <= slot < SLOT_COUNT for slot in falls):
        raise ValueError("falls must be global slots 0-9")
    lost = {1: 0, 2: 0}
    for slot in falls:
        lost[int(context.roster[slot].configured_team_id)] += 1
    after = list(runs)
    reached: list[tuple[int, int]] = []
    for team in (1, 2):
        if lost[team]:
            after[team - 1] = 0
            continue
        gained = lost[3 - team]
        if not gained:
            continue
        before = after[team - 1]
        after[team - 1] = before + gained
        mark = after[team - 1] // RUN_EVERY * RUN_EVERY
        if mark > before:
            reached.append((team, mark))
    return (after[0], after[1]), tuple(reached)


def match_moment_facts_v1(
    context: EvaluationEpisodeContext,
    *,
    fresh_start: bool,
    largest_lead_before: int,
    reached_runs: tuple[tuple[int, int], ...],
    rescues: tuple[RescueFact, ...],
) -> MatchMomentFactsV1:
    """Turn one view's counted facts into the presentation field's record.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        The recorded episode context; its roster gives public ID, team and
        class.
    fresh_start : bool, keyword-only
        Whether the match began at tick 0 with 0-0 (False on frame zero).
    largest_lead_before : int, keyword-only
        The largest lead at the start of this tick or earlier, 0 or more.
    reached_runs : tuple of (int, int), keyword-only
        (team ID, run) pairs from ``advance_team_runs_v1`` for this tick.
    rescues : tuple of (int, bool), keyword-only
        This view's (global slot, by_ultimate) rescues on this tick.

    Returns
    -------
    MatchMomentFactsV1
        The frame's ``match_moments`` field.

    Raises
    ------
    ValueError
        A run or rescue fails MatchMomentFactsV1's checks (pydantic's
        ValidationError).
    """
    return MatchMomentFactsV1(
        fresh_start=fresh_start,
        largest_lead_before=largest_lead_before,
        runs=tuple(
            TeamRunV1(
                team_id=cast("Literal[1, 2]", team),
                takedowns=run,
            )
            for team, run in reached_runs
        ),
        rescues=tuple(
            RescueV1(
                public_agent_id=context.roster[slot].public_agent_id,
                team_id=cast("Literal[1, 2]", context.roster[slot].configured_team_id),
                class_id=cast("Literal[1, 2, 3, 4, 5]", context.roster[slot].class_id),
                by_ultimate=by_ultimate,
            )
            for slot, by_ultimate in rescues
        ),
    )


def is_fresh_start_v1(simulator_step_count: int, scores: Iterable[int]) -> bool:
    """Return whether a match's first frame is a fresh start.

    Parameters
    ----------
    simulator_step_count : int
        The first frame's tick.
    scores : iterable of int
        The first frame's (Team A, Team B) scores.

    Returns
    -------
    bool
        True only for tick 0 with every score 0. Pure.
    """
    return simulator_step_count == 0 and all(int(score) == 0 for score in scores)


def score_lead_v1(scores: Iterable[int]) -> int:
    """Return the lead in one (Team A, Team B) score pair: the absolute difference.

    Parameters
    ----------
    scores : iterable of int
        Two scores, Team A then Team B.

    Returns
    -------
    int
        abs(Team A - Team B), 0 or more. Pure.
    """
    first, second = (int(score) for score in scores)
    return abs(first - second)


@dataclass(frozen=True, slots=True)
class MatchMomentHistoryV1:
    """Every view's Herald Cards history after one live frame, for the live session.

    The live service keeps one of these on each session. Oracle View has one
    row; Agent POV has one per possible recipient (global slots 0-9; an
    inactive slot keeps zeros), because the viewing champion can change at
    any time and each view must be exact the moment it is shown.

    Attributes
    ----------
    fresh_start : bool
        Whether the match began at tick 0 with 0-0; set by the first advance
        (from its start frame). Default False (frame zero).
    largest_lead : int
        The largest lead at the start of every tick advanced so far (0 or
        more), so on the current frame it is the fact
        ``largest_lead_before``. Default 0.
    oracle_runs : tuple of (int, int)
        Oracle View's runs after this frame (see ``advance_team_runs_v1``).
        Default ``NO_RUNS``.
    oracle_reached : tuple of (int, int)
        Oracle View's (team ID, run) pairs reached on this frame. Default
        empty.
    oracle_rescues : tuple of (int, bool)
        Oracle View's (global slot, by_ultimate) rescues on this frame.
        Default empty.
    agent_pov_runs : tuple of tuple of (int, int)
        Ten rows, one per recipient global slot, each that recipient's runs.
        Default all ``NO_RUNS``.
    agent_pov_reached : tuple of tuple of (int, int)
        Ten rows: the runs reached on this frame in each recipient's Agent
        POV. Default all empty.
    agent_pov_rescues : tuple of tuple of (int, bool)
        Ten rows: the rescues each recipient's team saw on this frame.
        Default all empty.

    Raises
    ------
    ValueError
        There are not ten recipient rows, a run row is not two whole numbers
        0 or more, or ``largest_lead`` is negative.
    """

    fresh_start: bool = False
    largest_lead: int = 0
    oracle_runs: tuple[int, int] = NO_RUNS
    oracle_reached: tuple[tuple[int, int], ...] = ()
    oracle_rescues: tuple[RescueFact, ...] = ()
    agent_pov_runs: tuple[tuple[int, int], ...] = (NO_RUNS,) * SLOT_COUNT
    agent_pov_reached: tuple[tuple[tuple[int, int], ...], ...] = ((),) * SLOT_COUNT
    agent_pov_rescues: tuple[tuple[RescueFact, ...], ...] = ((),) * SLOT_COUNT

    def __post_init__(self) -> None:
        """Check the recipient row counts, the run shapes and the lead."""
        if type(self.largest_lead) is not int or self.largest_lead < 0:
            raise ValueError("the largest lead must be a whole number 0 or more")
        rows = (self.agent_pov_runs, self.agent_pov_reached, self.agent_pov_rescues)
        if any(type(row) is not tuple or len(row) != SLOT_COUNT for row in rows):
            raise ValueError("Agent POV history rows need one row per global slot")
        for runs in (self.oracle_runs, *self.agent_pov_runs):
            if (
                type(runs) is not tuple
                or len(runs) != 2
                or any(type(value) is not int or value < 0 for value in runs)
            ):
                raise ValueError("team runs must be two whole numbers 0 or more")

    def advance(
        self,
        context: EvaluationEpisodeContext,
        *,
        start_frame_index: int,
        start_simulator_step_count: int,
        start_scores: tuple[int, int],
        oracle_falls: frozenset[int],
        oracle_rescues: tuple[RescueFact, ...],
        agent_pov_falls: tuple[frozenset[int], ...],
        agent_pov_rescues: tuple[tuple[RescueFact, ...], ...],
    ) -> MatchMomentHistoryV1:
        """Return every view's history one tick later.

        Parameters
        ----------
        context : EvaluationEpisodeContext
            The session's episode context (each slot's team).
        start_frame_index : int, keyword-only
            The transition's start frame index; 0 sets ``fresh_start``.
        start_simulator_step_count : int, keyword-only
            The start frame's tick.
        start_scores : tuple of (int, int), keyword-only
            The start frame's (Team A, Team B) scores; its lead joins
            ``largest_lead``.
        oracle_falls : frozenset of int, keyword-only
            Global slots of every recorded fall on the tick.
        oracle_rescues : tuple of (int, bool), keyword-only
            Oracle View's rescues (``oracle_tick_rescues_v1``).
        agent_pov_falls : tuple of frozenset of int, keyword-only
            Ten entries, one per recipient global slot: the falls that
            recipient's team saw (empty for an inactive slot or a tick
            without a fall).
        agent_pov_rescues : tuple of tuple of (int, bool), keyword-only
            Ten entries: the rescues each recipient's team saw
            (``agent_pov_tick_rescues_v1``; empty when none).

        Returns
        -------
        MatchMomentHistoryV1
            A new record; this one is unchanged.

        Raises
        ------
        ValueError
            ``agent_pov_falls`` or ``agent_pov_rescues`` does not have ten
            entries, or a fall is not a global slot 0-9.
        """
        if len(agent_pov_falls) != SLOT_COUNT or len(agent_pov_rescues) != SLOT_COUNT:
            raise ValueError("Agent POV ticks need one entry per global slot")
        oracle_runs, oracle_reached = advance_team_runs_v1(
            self.oracle_runs, context, oracle_falls
        )
        rows = tuple(
            advance_team_runs_v1(runs, context, falls)
            for runs, falls in zip(self.agent_pov_runs, agent_pov_falls, strict=True)
        )
        return MatchMomentHistoryV1(
            fresh_start=(
                is_fresh_start_v1(start_simulator_step_count, start_scores)
                if start_frame_index == 0
                else self.fresh_start
            ),
            largest_lead=max(self.largest_lead, score_lead_v1(start_scores)),
            oracle_runs=oracle_runs,
            oracle_reached=oracle_reached,
            oracle_rescues=oracle_rescues,
            agent_pov_runs=tuple(row[0] for row in rows),
            agent_pov_reached=tuple(row[1] for row in rows),
            agent_pov_rescues=agent_pov_rescues,
        )

    def facts(
        self,
        context: EvaluationEpisodeContext,
        recipient_global_slot: int | None,
    ) -> MatchMomentFactsV1:
        """Return one view's ``match_moments`` field for this frame.

        Parameters
        ----------
        context : EvaluationEpisodeContext
            The session's episode context (roster identities).
        recipient_global_slot : int or None
            None for Oracle View; otherwise the Agent POV recipient's global
            slot 0-9. Agent POV reads only its own recipient's rows.

        Returns
        -------
        MatchMomentFactsV1
            See ``match_moment_facts_v1``; runs and rescues are empty on
            most frames.
        """
        if recipient_global_slot is None:
            reached, rescues = self.oracle_reached, self.oracle_rescues
        else:
            reached = self.agent_pov_reached[recipient_global_slot]
            rescues = self.agent_pov_rescues[recipient_global_slot]
        return match_moment_facts_v1(
            context,
            fresh_start=self.fresh_start,
            largest_lead_before=self.largest_lead,
            reached_runs=reached,
            rescues=rescues,
        )


__all__ = [
    "NO_RUNS",
    "RUN_EVERY",
    "MatchMomentFactsV1",
    "MatchMomentHistoryV1",
    "RescueFact",
    "RescueV1",
    "TeamRunV1",
    "advance_team_runs_v1",
    "agent_pov_tick_rescues_v1",
    "is_fresh_start_v1",
    "match_moment_facts_v1",
    "oracle_tick_rescues_v1",
    "score_lead_v1",
]
