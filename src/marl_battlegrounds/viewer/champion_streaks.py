"""Count champion takedowns without falling for the Champion Streaks herald.

The Champion Streaks switch (``champion_streaks``, Moments group, off by
default) adds a herald card on the tick a champion takes part in its 5th,
10th, 15th or 20th (every 5th) takedown without falling, worded "10
Takedowns Without Falling". This module is the one owner of that count. The
browser never counts: it draws the presentation field ``champion_streaks`` it
receives.

Who calls it:

- The Replay Viewer service (``viewer.replay_service``) counts once per
  replay and view, frame by frame, and reuses the table across requests,
  seeks and reloads of the page.
- The live BattleClient and DevClient service carries every view's counts on
  its session (``_battle_client.model.DebuggerSession.champion_takedowns``),
  advanced once per transition by
  ``_battle_client.live_presentation.advance_live_champion_takedowns_v1``.
  The same record keeps each view's ticks with a fall from the Fallen panel's
  window, so live frames carry ``recent_falls`` exactly like replay frames.

The rule, per view (one "audience"):

- A takedown is one fall plus one champion that took part in it. Each fall a
  champion took part in adds 1 to its count; kill credit is shared, so two
  champions can both gain from the same fall.
- A champion's count goes back to 0 when it falls. A champion that falls on
  the same tick gains nothing from that tick.
- A streak is reached on the tick a count first passes a multiple of 5
  (``STREAK_EVERY``): 5, 10, 15, 20 and so on.
- Oracle View counts every recorded fall, with the champions that dealt
  damage on the lethal transition (``lethal_damage_contribution`` events), the
  same killers the Oracle Fallen feed names. Priest healing support is not a
  takedown here.
- Agent POV counts only what the viewing champion's team saw, read from that
  recipient's own Agent POV visual events (never from Oracle facts): a fall
  the team saw (an ``agent_died`` row), and a damaging Basic or Ultimate on
  that fallen champion from a source the team saw at the start of the tick
  (an ``ability_activated`` row), the same killers the Agent POV Fallen feed
  names. Only a fall the team saw resets a count.

The same per-tick takedowns also feed the Fallen panel's earlier rows
(``recent_falls_v1``, the presentation field ``recent_falls``): each fall
with its killers, its points (2 for a home fall, else 1) and the damaging
Ultimate caster, so the browser never keeps that history itself. A tick's
TickTakedownsV1 carries the extra facts for it (``ultimates`` and
``home_falls``).

Everything is plain Python on small integers: no JAX, file, network or
simulator work.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Annotated, Final, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marl_battlegrounds.evaluation._recording.events import (
    team_deathmatch_home_fall_slots_v1,
)
from marl_battlegrounds.evaluation._replays.wire_shapes import MAX_AGENT_SLOTS_V1
from marl_battlegrounds.evaluation.models import (
    AbilityActivatedEventV1,
    AgentDiedEventV1,
    LethalDamageContributionEventV1,
)
from marl_battlegrounds.rendering.scene import (
    AbilityActivatedEventV2,
    AgentDiedEventV2,
    LethalDamageContributionEventV2,
)
from marl_battlegrounds.rendering.vocabulary import (
    HUNTER_CLASS_ID,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.models import EvaluationEpisodeContext
    from marl_battlegrounds.rendering.authorized_presentation import (
        AgentPovVisualIncomingSummaryV1,
    )

SLOT_COUNT: Final = MAX_AGENT_SLOTS_V1
"""Global champion slots (10: 0 to 4 Team A, 5 to 9 Team B); counts have this length."""

STREAK_EVERY: Final = 5
"""A champion's streak is announced at every multiple of this many takedowns."""

DAMAGING_ULTIMATE_CLASS_IDS: Final = frozenset(
    (WARRIOR_CLASS_ID, HUNTER_CLASS_ID, ROGUE_CLASS_ID)
)
"""Classes whose Ultimate (Charge, Freezing Trap, Crippling Poison) deals damage.

Mage Burst only strengthens the Mage and Priest Holy Word: Salvation heals, so
neither names a killer. Every Basic deals damage except the Priest's, which
heals."""

ZERO_COUNTS: Final[tuple[int, ...]] = (0,) * SLOT_COUNT
"""Counts before any takedown: one 0 per global slot."""

FEED_WINDOW_TICKS: Final = 8
"""The Fallen panel lists the falls of this many ticks: the shown tick and the
7 before it. Must equal FEED_WINDOW_TICKS in web/visual_debugger/src/looks/
requiems.js."""


class ChampionStreakV1(BaseModel):
    """One champion that reached a streak on this tick, as the browser receives it.

    Attributes
    ----------
    public_agent_id : str
        The champion's public ID, as in the presentation's scene and events.
    team_id : {1, 2}
        The champion's team (1 Team A, 2 Team B).
    class_id : {1, 2, 3, 4, 5}
        The champion's class (Mage, Warrior, Hunter, Rogue, Priest).
    takedowns : int
        The streak reached on this tick: the champion has now taken part in
        this many takedowns without falling. A multiple of ``STREAK_EVERY``
        (5, 10, 15, 20 and so on).

    Raises
    ------
    pydantic.ValidationError
        ``takedowns`` is below 5 or not a multiple of 5, or another field is
        outside its values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    public_agent_id: Annotated[str, Field(min_length=1)]
    team_id: Literal[1, 2]
    class_id: Literal[1, 2, 3, 4, 5]
    takedowns: Annotated[int, Field(ge=STREAK_EVERY)]

    @field_validator("takedowns")
    @classmethod
    def _every_fifth(cls, takedowns: int) -> int:
        """Accept only a multiple of ``STREAK_EVERY``; return it unchanged."""
        if takedowns % STREAK_EVERY:
            raise ValueError(f"a streak must be a multiple of {STREAK_EVERY}")
        return takedowns


@dataclass(frozen=True, slots=True)
class TickTakedownsV1:
    """The falls and takedowns one view saw on one tick, by global slot.

    Attributes
    ----------
    falls : frozenset of int
        Global slots (0-9) of the champions this view saw fall on the tick.
    takedowns : frozenset of (int, int)
        (source, fallen) global-slot pairs: the source took part in that
        fall. Every fallen slot is in ``falls``; source and fallen champion
        are on opposite teams.
    ultimates : frozenset of (int, int), default empty
        (caster, fallen) pairs: the first damaging Ultimate (Charge, Freezing
        Trap or Crippling Poison) that struck that fallen champion on the
        tick, in event order, at most one per fall. Each pair is also in
        ``takedowns``. Empty when only Basics did it, and for producers that
        do not read Ultimates.
    home_falls : frozenset of int or None, default None
        The fallen slots whose fall scored 2 points (a fall in its own Red
        Zone), a subset of ``falls``; the rest scored 1. None means unknown
        (the producer did not read it), and the points are then unknown too.

    Raises
    ------
    TypeError
        A slot is not a Python int or a field has the wrong container.
    ValueError
        A slot is outside 0-9, a takedown names a fall that is not in
        ``falls``, a champion takes down itself, an Ultimate pair is not a
        takedown or repeats a fall, or a home fall is not in ``falls``.
    """

    falls: frozenset[int]
    takedowns: frozenset[tuple[int, int]]
    ultimates: frozenset[tuple[int, int]] = frozenset()
    home_falls: frozenset[int] | None = None

    def __post_init__(self) -> None:
        """Check slot ranges and that every takedown joins a fall in ``falls``."""
        if (
            type(self.falls) is not frozenset
            or type(self.takedowns) is not frozenset
            or type(self.ultimates) is not frozenset
            or (self.home_falls is not None and type(self.home_falls) is not frozenset)
        ):
            raise TypeError("tick takedowns must hold frozensets")
        slots = (
            *self.falls,
            *(slot for pair in self.takedowns for slot in pair),
            *(slot for pair in self.ultimates for slot in pair),
            *(self.home_falls or ()),
        )
        if any(type(slot) is not int for slot in slots):
            raise TypeError("tick takedown slots must be Python ints")
        if any(not 0 <= slot < SLOT_COUNT for slot in slots):
            raise ValueError("tick takedown slots must be global slots 0-9")
        for source, fallen in self.takedowns:
            if fallen not in self.falls or source == fallen:
                raise ValueError("a takedown must join another champion's fall")
        fallen_by_ultimate = [fallen for _caster, fallen in self.ultimates]
        if not self.ultimates <= self.takedowns or len(set(fallen_by_ultimate)) != len(
            fallen_by_ultimate
        ):
            raise ValueError("an Ultimate must be one takedown per fall")
        if self.home_falls is not None and not self.home_falls <= self.falls:
            raise ValueError("a home fall must be a fall of the tick")


NO_TAKEDOWNS: Final = TickTakedownsV1(falls=frozenset(), takedowns=frozenset())
"""A tick on which this view saw no fall."""


def oracle_tick_takedowns_v1(
    context: EvaluationEpisodeContext,
    events: Iterable[object],
    *,
    start_positions: Sequence[Sequence[float]] | None = None,
) -> TickTakedownsV1:
    """Read Oracle View's falls and takedowns from one transition's recorded events.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        The recorded episode context; its roster gives each slot's team and
        class.
    events : iterable of object
        The recorded events of one transition: evaluation events (live
        ``EvaluationTransitionV1.events`` or a replay's) or the events of a
        canonical visual batch. Only agent_died, lethal_damage_contribution
        and ability_activated events (V1 or V2) are read; others are ignored.
    start_positions : sequence of (x, y) or None, optional, keyword-only
        Every global slot's position at the transition start (usually the
        start frame's ``snapshot.agent_positions``), for the home-fall rule
        (``evaluation._recording.events.team_deathmatch_home_fall_slots_v1``). None (the
        default) leaves ``home_falls`` None (points unknown).

    Returns
    -------
    TickTakedownsV1
        Every recorded fall; every (source, fallen) pair from a lethal damage
        contribution on an opposing champion that fell this tick; for each
        fall, the first damaging Ultimate on it (in event order) whose caster
        is one of those contributors (the browser's Fallen feed and Ultimate
        Takedown rule); and the home falls when start_positions is given.

    Notes
    -----
    Pure; reads the events, the roster and the given positions only.
    """
    rows = tuple(events)
    falls = frozenset(
        event.recipient_global_slot
        for event in rows
        if isinstance(event, (AgentDiedEventV1, AgentDiedEventV2))
    )
    takedowns = frozenset(
        (event.source_global_slot, event.recipient_global_slot)
        for event in rows
        if isinstance(
            event, (LethalDamageContributionEventV1, LethalDamageContributionEventV2)
        )
        and event.recipient_global_slot in falls
        and _opposing(context, event.source_global_slot, event.recipient_global_slot)
    )
    ultimates: dict[int, int] = {}
    for event in rows:
        if (
            isinstance(event, (AbilityActivatedEventV1, AbilityActivatedEventV2))
            and event.ability_component == "ultimate"
            and event.recipient_global_slot is not None
            and event.recipient_global_slot not in ultimates
            and (event.source_global_slot, event.recipient_global_slot) in takedowns
            and context.roster[event.source_global_slot].class_id
            in DAMAGING_ULTIMATE_CLASS_IDS
        ):
            ultimates[event.recipient_global_slot] = event.source_global_slot
    home_falls = (
        None
        if start_positions is None
        else team_deathmatch_home_fall_slots_v1(
            context,
            start_positions,
            tuple(slot in falls for slot in range(SLOT_COUNT)),
        )
        & falls
    )
    return TickTakedownsV1(
        falls=falls,
        takedowns=takedowns,
        ultimates=frozenset((caster, fallen) for fallen, caster in ultimates.items()),
        home_falls=home_falls,
    )


def agent_pov_tick_takedowns_v1(
    context: EvaluationEpisodeContext,
    visual_events: AgentPovVisualIncomingSummaryV1,
) -> TickTakedownsV1:
    """Read what one Agent POV recipient's team saw fall and strike on one tick.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        The recorded episode context; its roster maps public IDs to global
        slots and teams.
    visual_events : AgentPovVisualIncomingSummaryV1
        That recipient's own Agent POV visual events for the transition, as
        ``rendering.authorized_presentation.
        build_agent_pov_visual_incoming_summary_v1`` builds them (the same
        filter the arena uses).

    Returns
    -------
    TickTakedownsV1
        The falls the team saw (``agent_died`` rows) and, for each, the
        sources of damaging Basic or Ultimate activations on that fallen
        champion (``ability_activated`` rows, whose source the team saw at
        the start of the tick). A Basic is damaging unless its source is a
        Priest; an Ultimate is damaging for ``DAMAGING_ULTIMATE_CLASS_IDS``.
        ``ultimates`` names, per fall, the first damaging Ultimate on it in
        event order; ``home_falls`` holds the seen falls whose row carries
        home_fall True (a seen fall always carries the flag).

    Notes
    -----
    Pure. It reads only the recipient's own visual events and the fixed
    roster, never an Oracle event, so an unseen fall or blow adds nothing.
    """
    slot_by_public_id = {row.public_agent_id: row.global_slot for row in context.roster}
    class_by_public_id = {
        row.agent_public_agent_id: row.agent_class_id
        for row in visual_events.agent_phase_trajectories
    }
    falls = frozenset(
        slot_by_public_id[event.recipient_anchor.public_agent_id]
        for event in visual_events.events
        if event.event_kind == "agent_died"
    )
    home_falls = frozenset(
        slot_by_public_id[event.recipient_anchor.public_agent_id]
        for event in visual_events.events
        if event.event_kind == "agent_died" and getattr(event, "home_fall", False)
    )
    takedowns: set[tuple[int, int]] = set()
    ultimates: dict[int, int] = {}
    for event in visual_events.events:
        if event.event_kind != "ability_activated" or event.recipient_anchor is None:
            continue
        source_id = event.source_anchor.public_agent_id
        fallen = slot_by_public_id[event.recipient_anchor.public_agent_id]
        class_id = class_by_public_id[source_id]
        ultimate = event.ability_component == "ultimate"
        damaging = (
            class_id in DAMAGING_ULTIMATE_CLASS_IDS
            if ultimate
            else class_id != PRIEST_CLASS_ID
        )
        source = slot_by_public_id[source_id]
        if damaging and fallen in falls and _opposing(context, source, fallen):
            takedowns.add((source, fallen))
            if ultimate and fallen not in ultimates:
                ultimates[fallen] = source
    return TickTakedownsV1(
        falls=falls,
        takedowns=frozenset(takedowns),
        ultimates=frozenset((caster, fallen) for fallen, caster in ultimates.items()),
        home_falls=home_falls,
    )


def advance_takedown_counts_v1(
    counts: tuple[int, ...], tick: TickTakedownsV1
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Add one tick to one view's counts of takedowns without falling.

    Parameters
    ----------
    counts : tuple of int
        Ten counts in global slot order (0 to 4 Team A, 5 to 9 Team B), each
        0 or more: takedowns taken part in since the champion last fell (in
        this view). Use ``ZERO_COUNTS`` on frame zero.
    tick : TickTakedownsV1
        What this view saw on the next tick.

    Returns
    -------
    counts : tuple of int
        The ten counts after the tick. A source gains 1 per fall it took part
        in, unless it fell this tick too; every fallen champion is then 0.
    reached : tuple of (int, int)
        (global slot, streak) for every champion whose count first passed a
        multiple of ``STREAK_EVERY`` on this tick, by slot, with the largest
        multiple passed (a tick adds at most 5, one per enemy, so it passes at
        most one). Empty on most ticks.

    Raises
    ------
    ValueError
        ``counts`` is not ten whole numbers 0 or more.
    """
    _require_counts(counts)
    after = list(counts)
    reached: list[tuple[int, int]] = []
    gained = Counter(
        source for source, _fallen in tick.takedowns if source not in tick.falls
    )
    for source in sorted(gained):
        before = after[source]
        after[source] = before + gained[source]
        mark = after[source] // STREAK_EVERY * STREAK_EVERY
        if mark > before:
            reached.append((source, mark))
    for fallen in tick.falls:
        after[fallen] = 0
    return tuple(after), tuple(reached)


def champion_streaks_v1(
    context: EvaluationEpisodeContext, reached: tuple[tuple[int, int], ...]
) -> tuple[ChampionStreakV1, ...]:
    """Turn one view's reached streaks into the presentation field's records.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        The recorded episode context; its roster gives public ID, team and
        class.
    reached : tuple of (int, int)
        (global slot, streak) pairs from ``advance_takedown_counts_v1`` for
        the shown frame.

    Returns
    -------
    tuple of ChampionStreakV1
        One record per champion, larger streak first, then by global slot.
        Empty when nobody reached a streak on this tick.

    Raises
    ------
    ValueError
        A streak is not a multiple of 5 of at least 5 (pydantic's
        ValidationError).
    """
    ordered = sorted(reached, key=lambda pair: (-pair[1], pair[0]))
    return tuple(
        ChampionStreakV1(
            public_agent_id=context.roster[slot].public_agent_id,
            team_id=cast("Literal[1, 2]", context.roster[slot].configured_team_id),
            class_id=cast("Literal[1, 2, 3, 4, 5]", context.roster[slot].class_id),
            takedowns=takedowns,
        )
        for slot, takedowns in ordered
    )


@dataclass(frozen=True, slots=True)
class ChampionTakedownsV1:
    """Every view's takedown counts after one live frame, for the live session.

    The live service keeps one of these on each session. Oracle View has one
    row of counts; Agent POV has one per possible recipient (global slots
    0-9; an inactive slot keeps zeros), because the viewing champion can
    change at any time and each view must be exact the moment it is shown.
    The record also keeps each view's recent ticks with a fall, the Fallen
    panel's earlier rows (``recent_falls``), for the same reason: a view
    switch or a reload of the page shows the same rows as watching that view
    all along.

    Attributes
    ----------
    oracle_counts : tuple of int
        Oracle View's ten counts after this frame (see
        ``advance_takedown_counts_v1``). Default ``ZERO_COUNTS``.
    oracle_reached : tuple of (int, int)
        Oracle View's (global slot, streak) pairs reached on this frame.
        Default empty.
    agent_pov_counts : tuple of tuple of int
        Ten rows, one per recipient global slot, each that recipient's ten
        counts. Default all zeros.
    agent_pov_reached : tuple of tuple of (int, int)
        Ten rows, one per recipient: the streaks reached on this frame in
        that recipient's Agent POV. Default all empty.
    oracle_recent : tuple of (int, TickTakedownsV1)
        Oracle View's ticks with a fall that a later frame's Fallen panel may
        still list: (simulator tick of the frame the fall reached, what the
        view saw on it), oldest first, only ticks with at least one fall,
        none older than ``FEED_WINDOW_TICKS - 1`` ticks before the newest
        advance. Default empty. Read it through ``recent_falls``.
    agent_pov_recent : tuple of tuple of (int, TickTakedownsV1)
        Ten rows, one per recipient global slot, each that recipient's Agent
        POV ticks with a fall, kept like ``oracle_recent``. Default all empty.

    Raises
    ------
    ValueError
        A row does not have ten whole counts 0 or more, there are not ten
        recipient rows, a reached streak is not a count's passed value, or a
        recent row is not (tick 0 or more, TickTakedownsV1 with a fall) in
        rising tick order.
    """

    oracle_counts: tuple[int, ...] = ZERO_COUNTS
    oracle_reached: tuple[tuple[int, int], ...] = ()
    agent_pov_counts: tuple[tuple[int, ...], ...] = (ZERO_COUNTS,) * SLOT_COUNT
    agent_pov_reached: tuple[tuple[tuple[int, int], ...], ...] = ((),) * SLOT_COUNT
    oracle_recent: tuple[tuple[int, TickTakedownsV1], ...] = ()
    agent_pov_recent: tuple[tuple[tuple[int, TickTakedownsV1], ...], ...] = (
        (),
    ) * SLOT_COUNT

    def __post_init__(self) -> None:
        """Check every row's shape, each reached streak and each recent row."""
        if (
            type(self.agent_pov_counts) is not tuple
            or type(self.agent_pov_reached) is not tuple
            or type(self.agent_pov_recent) is not tuple
            or len(self.agent_pov_counts) != SLOT_COUNT
            or len(self.agent_pov_reached) != SLOT_COUNT
            or len(self.agent_pov_recent) != SLOT_COUNT
        ):
            raise ValueError("Agent POV takedown rows need one row per global slot")
        for recent in (self.oracle_recent, *self.agent_pov_recent):
            if (
                type(recent) is not tuple
                or any(
                    type(row) is not tuple
                    or len(row) != 2
                    or type(row[0]) is not int
                    or row[0] < 0
                    or type(row[1]) is not TickTakedownsV1
                    or not row[1].falls
                    for row in recent
                )
                or any(a[0] >= b[0] for a, b in pairwise(recent))
            ):
                raise ValueError(
                    "recent takedown rows must be rising (tick, fall) pairs"
                )
        for counts, reached in (
            (self.oracle_counts, self.oracle_reached),
            *zip(self.agent_pov_counts, self.agent_pov_reached, strict=True),
        ):
            _require_counts(counts)
            if type(reached) is not tuple or any(
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not int
                or not 0 <= pair[0] < SLOT_COUNT
                or type(pair[1]) is not int
                or pair[1] < STREAK_EVERY
                or pair[1] % STREAK_EVERY
                or counts[pair[0]] < pair[1]
                for pair in reached
            ):
                raise ValueError("a reached streak must join its champion's count")

    def advance(
        self,
        oracle: TickTakedownsV1,
        agent_pov: tuple[TickTakedownsV1, ...],
        *,
        tick: int,
    ) -> ChampionTakedownsV1:
        """Return every view's counts and recent falls one tick later.

        Parameters
        ----------
        oracle : TickTakedownsV1
            What Oracle View saw on the tick (``oracle_tick_takedowns_v1``).
        agent_pov : tuple of TickTakedownsV1
            Ten entries, one per recipient global slot, each what that
            recipient's Agent POV saw (``agent_pov_tick_takedowns_v1``);
            ``NO_TAKEDOWNS`` for an inactive slot or a tick without a fall.
        tick : int, keyword-only
            The simulator tick of the frame the transition reached (the
            successor frame's ``simulator_step_count``), 0 or more and later
            than every tick already kept.

        Returns
        -------
        ChampionTakedownsV1
            A new record; this one is unchanged. Each view that saw a fall
            adds ``(tick, its TickTakedownsV1)`` to its recent rows, and every
            view drops rows older than ``tick - FEED_WINDOW_TICKS + 1`` (no
            later frame's Fallen panel can list them).

        Raises
        ------
        ValueError
            ``agent_pov`` does not have ten entries, or a view that saw a fall
            at ``tick`` already keeps a row at ``tick`` or later (each view's
            kept rows must rise). A tick without a fall adds no row, so it is
            not checked.
        """
        if len(agent_pov) != SLOT_COUNT:
            raise ValueError("Agent POV ticks need one entry per global slot")
        oracle_counts, oracle_reached = advance_takedown_counts_v1(
            self.oracle_counts, oracle
        )
        rows = tuple(
            advance_takedown_counts_v1(counts, seen)
            for counts, seen in zip(self.agent_pov_counts, agent_pov, strict=True)
        )
        oldest = tick - FEED_WINDOW_TICKS + 1

        def kept(
            recent: tuple[tuple[int, TickTakedownsV1], ...], seen: TickTakedownsV1
        ) -> tuple[tuple[int, TickTakedownsV1], ...]:
            # Keep the rows a later frame may still list, then add this tick's.
            return tuple(row for row in recent if row[0] >= oldest) + (
                ((tick, seen),) if seen.falls else ()
            )

        return ChampionTakedownsV1(
            oracle_counts=oracle_counts,
            oracle_reached=oracle_reached,
            agent_pov_counts=tuple(row[0] for row in rows),
            agent_pov_reached=tuple(row[1] for row in rows),
            oracle_recent=kept(self.oracle_recent, oracle),
            agent_pov_recent=tuple(
                kept(recent, seen)
                for recent, seen in zip(self.agent_pov_recent, agent_pov, strict=True)
            ),
        )

    def quiet(self) -> ChampionTakedownsV1:
        """Return the record for a tick on which nobody fell.

        Counts and recent rows stay the same and nothing is reached; this
        skips all Agent POV work on the common tick without a fall (a recent
        row outside the window is never listed, see ``recent_falls``, and is
        dropped by the next ``advance``).

        Returns
        -------
        ChampionTakedownsV1
            ``self`` when nothing was reached on this frame; otherwise a new
            record with the same counts and recent rows and no reached
            streaks.
        """
        if not self.oracle_reached and not any(self.agent_pov_reached):
            return self
        return ChampionTakedownsV1(
            oracle_counts=self.oracle_counts,
            agent_pov_counts=self.agent_pov_counts,
            oracle_recent=self.oracle_recent,
            agent_pov_recent=self.agent_pov_recent,
        )

    def recent_falls(
        self,
        context: EvaluationEpisodeContext,
        recipient_global_slot: int | None,
        tick: int,
    ) -> tuple[RecentFallV1, ...]:
        """Return one view's ``recent_falls`` field for the frame at ``tick``.

        Parameters
        ----------
        context : EvaluationEpisodeContext
            The session's episode context (roster public IDs).
        recipient_global_slot : int or None
            None for Oracle View; otherwise the Agent POV recipient's global
            slot 0-9. Agent POV reads only its own recipient's rows.
        tick : int
            The shown frame's simulator tick (its ``simulator_step_count``).

        Returns
        -------
        tuple of RecentFallV1
            The falls this view saw on the 7 ticks before ``tick`` (``tick -
            FEED_WINDOW_TICKS < row tick < tick``), oldest first, as
            ``recent_falls_v1`` builds them; the shown tick's own falls are
            left out (the browser's plan draws them at the Fall beat), the
            same window the Replay Viewer service uses. Empty when none.
        """
        recent = (
            self.oracle_recent
            if recipient_global_slot is None
            else self.agent_pov_recent[recipient_global_slot]
        )
        return recent_falls_v1(
            context,
            [row for row in recent if tick - FEED_WINDOW_TICKS < row[0] < tick],
        )

    def streaks(
        self,
        context: EvaluationEpisodeContext,
        recipient_global_slot: int | None,
    ) -> tuple[ChampionStreakV1, ...]:
        """Return one view's ``champion_streaks`` field for this frame.

        Parameters
        ----------
        context : EvaluationEpisodeContext
            The session's episode context (roster identities).
        recipient_global_slot : int or None
            None for Oracle View; otherwise the Agent POV recipient's global
            slot 0-9. Agent POV reads only its own recipient's row.

        Returns
        -------
        tuple of ChampionStreakV1
            See ``champion_streaks_v1``; empty on most frames.
        """
        reached = (
            self.oracle_reached
            if recipient_global_slot is None
            else self.agent_pov_reached[recipient_global_slot]
        )
        return champion_streaks_v1(context, reached)


class RecentFallV1(BaseModel):
    """One fall of an earlier tick still inside the Fallen panel's window.

    The presentation field ``recent_falls`` carries these for the ticks
    before the shown one, so the panel shows the same rows after a seek, a
    reload or a view switch as after playing up to the tick. The shown tick's
    own falls come from the browser's plan, which times them to the Fall beat.

    Attributes
    ----------
    tick : int
        The simulator tick of the frame the fall reached (the frame's
        ``simulator_step_count``), 0 or more.
    victim_public_agent_id : str
        The fallen champion's public ID.
    killer_public_agent_ids : tuple of str
        The champions this view credits with the fall, by global slot: Oracle
        View the lethal damage contributors; Agent POV only those whose
        damaging blow the team saw land (a source seen at the start of the
        tick, including one that then left sight). Empty when nobody is
        named. An old recording with no lethal damage contribution events
        names nobody here (the browser's plan falls back to the match
        summary only for the shown tick).
    points : {1, 2} or None
        2 for a home fall (in the fallen champion's own Red Zone), else 1;
        None when unknown.
    ultimate_public_agent_id : str or None
        The caster of the first damaging Ultimate that struck the fallen
        champion on that tick (one of the killers), or None when only Basics
        did it.

    Raises
    ------
    pydantic.ValidationError
        A field is outside its values, a killer repeats or is the victim, or
        the Ultimate caster is not a killer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tick: Annotated[int, Field(ge=0)]
    victim_public_agent_id: Annotated[str, Field(min_length=1)]
    killer_public_agent_ids: tuple[Annotated[str, Field(min_length=1)], ...]
    points: Literal[1, 2] | None
    ultimate_public_agent_id: Annotated[str, Field(min_length=1)] | None

    @field_validator("killer_public_agent_ids")
    @classmethod
    def _unique_killers(cls, killers: tuple[str, ...]) -> tuple[str, ...]:
        """Accept each killer once; return the tuple unchanged."""
        if len(set(killers)) != len(killers):
            raise ValueError("a recent fall names each killer once")
        return killers

    @model_validator(mode="after")
    def _join_killers(self) -> Self:
        """Require the victim to be no killer and the caster to be a killer."""
        if self.victim_public_agent_id in self.killer_public_agent_ids or (
            self.ultimate_public_agent_id is not None
            and self.ultimate_public_agent_id not in self.killer_public_agent_ids
        ):
            raise ValueError("a recent fall's killers and caster do not join")
        return self


def recent_falls_v1(
    context: EvaluationEpisodeContext,
    ticks: Sequence[tuple[int, TickTakedownsV1]],
) -> tuple[RecentFallV1, ...]:
    """Turn one view's earlier ticks into the ``recent_falls`` field's records.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        The recorded episode context; its roster maps global slots to public
        IDs.
    ticks : sequence of (int, TickTakedownsV1)
        (tick, what this view saw on it) for the ticks to list, oldest first:
        the caller passes the ticks before the shown one still inside the
        window (shown tick - FEED_WINDOW_TICKS + 1 to shown tick - 1).

    Returns
    -------
    tuple of RecentFallV1
        One record per fall, by tick, then by the fallen champion's global
        slot. Killers are in global slot order. points come from
        ``home_falls`` (None when it is None). Empty when no tick had a fall.

    Notes
    -----
    Pure. The view's rules (Oracle View or Agent POV) are already in each
    TickTakedownsV1 (``oracle_tick_takedowns_v1`` or
    ``agent_pov_tick_takedowns_v1``), so this adds no rule of its own.
    """
    public_by_slot = {row.global_slot: row.public_agent_id for row in context.roster}
    rows: list[RecentFallV1] = []
    for tick, takedowns in ticks:
        caster_by_fallen = {fallen: caster for caster, fallen in takedowns.ultimates}
        for fallen in sorted(takedowns.falls):
            killers = sorted(
                source for source, victim in takedowns.takedowns if victim == fallen
            )
            caster = caster_by_fallen.get(fallen)
            home = takedowns.home_falls
            rows.append(
                RecentFallV1(
                    tick=tick,
                    victim_public_agent_id=public_by_slot[fallen],
                    killer_public_agent_ids=tuple(
                        public_by_slot[slot] for slot in killers
                    ),
                    points=None if home is None else 2 if fallen in home else 1,
                    ultimate_public_agent_id=(
                        None if caster is None else public_by_slot[caster]
                    ),
                )
            )
    return tuple(rows)


def _opposing(context: EvaluationEpisodeContext, source: int, fallen: int) -> bool:
    """Return whether two global slots are on different configured teams."""
    return (
        context.roster[source].configured_team_id
        != context.roster[fallen].configured_team_id
    )


def _require_counts(counts: object) -> None:
    """Raise ValueError unless counts is a tuple of ten Python ints 0 or more."""
    values = cast("tuple[object, ...]", counts)
    if (
        type(counts) is not tuple
        or len(values) != SLOT_COUNT
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise ValueError("takedown counts must be ten whole numbers 0 or more")


__all__ = [
    "DAMAGING_ULTIMATE_CLASS_IDS",
    "FEED_WINDOW_TICKS",
    "NO_TAKEDOWNS",
    "STREAK_EVERY",
    "ZERO_COUNTS",
    "ChampionStreakV1",
    "ChampionTakedownsV1",
    "RecentFallV1",
    "TickTakedownsV1",
    "advance_takedown_counts_v1",
    "agent_pov_tick_takedowns_v1",
    "champion_streaks_v1",
    "oracle_tick_takedowns_v1",
    "recent_falls_v1",
]
