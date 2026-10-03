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

Everything is plain Python on small integers: no JAX, file, network or
simulator work.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from marl_battlegrounds.evaluation.models import (
    AgentDiedEventV1,
    LethalDamageContributionEventV1,
)
from marl_battlegrounds.evaluation.wire_shapes import MAX_AGENT_SLOTS_V1
from marl_battlegrounds.rendering.scene import (
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

    Raises
    ------
    TypeError
        A slot is not a Python int or a field has the wrong container.
    ValueError
        A slot is outside 0-9, a takedown names a fall that is not in
        ``falls``, or a champion takes down itself.
    """

    falls: frozenset[int]
    takedowns: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        """Check slot ranges and that every takedown joins a fall in ``falls``."""
        if type(self.falls) is not frozenset or type(self.takedowns) is not frozenset:
            raise TypeError("tick takedowns must hold frozensets")
        slots = (*self.falls, *(slot for pair in self.takedowns for slot in pair))
        if any(type(slot) is not int for slot in slots):
            raise TypeError("tick takedown slots must be Python ints")
        if any(not 0 <= slot < SLOT_COUNT for slot in slots):
            raise ValueError("tick takedown slots must be global slots 0-9")
        for source, fallen in self.takedowns:
            if fallen not in self.falls or source == fallen:
                raise ValueError("a takedown must join another champion's fall")


NO_TAKEDOWNS: Final = TickTakedownsV1(falls=frozenset(), takedowns=frozenset())
"""A tick on which this view saw no fall."""


def oracle_tick_takedowns_v1(
    context: EvaluationEpisodeContext, events: Iterable[object]
) -> TickTakedownsV1:
    """Read Oracle View's falls and takedowns from one transition's recorded events.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        The recorded episode context; its roster gives each slot's team.
    events : iterable of object
        The recorded events of one transition: evaluation events (live
        ``EvaluationTransitionV1.events`` or a replay's) or the events of a
        canonical visual batch. Only agent_died and lethal_damage_contribution
        events (V1 or V2) are read; others are ignored.

    Returns
    -------
    TickTakedownsV1
        Every recorded fall, and every (source, fallen) pair from a lethal
        damage contribution on an opposing champion that fell this tick.

    Notes
    -----
    Pure; reads the events only.
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
    return TickTakedownsV1(falls=falls, takedowns=takedowns)


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
    takedowns: set[tuple[int, int]] = set()
    for event in visual_events.events:
        if event.event_kind != "ability_activated" or event.recipient_anchor is None:
            continue
        source_id = event.source_anchor.public_agent_id
        fallen = slot_by_public_id[event.recipient_anchor.public_agent_id]
        class_id = class_by_public_id[source_id]
        damaging = (
            class_id in DAMAGING_ULTIMATE_CLASS_IDS
            if event.ability_component == "ultimate"
            else class_id != PRIEST_CLASS_ID
        )
        source = slot_by_public_id[source_id]
        if damaging and fallen in falls and _opposing(context, source, fallen):
            takedowns.add((source, fallen))
    return TickTakedownsV1(falls=falls, takedowns=frozenset(takedowns))


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

    Raises
    ------
    ValueError
        A row does not have ten whole counts 0 or more, there are not ten
        recipient rows, or a reached streak is not a count's passed value.
    """

    oracle_counts: tuple[int, ...] = ZERO_COUNTS
    oracle_reached: tuple[tuple[int, int], ...] = ()
    agent_pov_counts: tuple[tuple[int, ...], ...] = (ZERO_COUNTS,) * SLOT_COUNT
    agent_pov_reached: tuple[tuple[tuple[int, int], ...], ...] = ((),) * SLOT_COUNT

    def __post_init__(self) -> None:
        """Check every row's shape and that each reached streak joins its count."""
        if (
            type(self.agent_pov_counts) is not tuple
            or type(self.agent_pov_reached) is not tuple
            or len(self.agent_pov_counts) != SLOT_COUNT
            or len(self.agent_pov_reached) != SLOT_COUNT
        ):
            raise ValueError("Agent POV takedown rows need one row per global slot")
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
    ) -> ChampionTakedownsV1:
        """Return every view's counts one tick later.

        Parameters
        ----------
        oracle : TickTakedownsV1
            What Oracle View saw on the tick (``oracle_tick_takedowns_v1``).
        agent_pov : tuple of TickTakedownsV1
            Ten entries, one per recipient global slot, each what that
            recipient's Agent POV saw (``agent_pov_tick_takedowns_v1``);
            ``NO_TAKEDOWNS`` for an inactive slot or a tick without a fall.

        Returns
        -------
        ChampionTakedownsV1
            A new record; this one is unchanged.

        Raises
        ------
        ValueError
            ``agent_pov`` does not have ten entries.
        """
        if len(agent_pov) != SLOT_COUNT:
            raise ValueError("Agent POV ticks need one entry per global slot")
        oracle_counts, oracle_reached = advance_takedown_counts_v1(
            self.oracle_counts, oracle
        )
        rows = tuple(
            advance_takedown_counts_v1(counts, tick)
            for counts, tick in zip(self.agent_pov_counts, agent_pov, strict=True)
        )
        return ChampionTakedownsV1(
            oracle_counts=oracle_counts,
            oracle_reached=oracle_reached,
            agent_pov_counts=tuple(row[0] for row in rows),
            agent_pov_reached=tuple(row[1] for row in rows),
        )

    def quiet(self) -> ChampionTakedownsV1:
        """Return the record for a tick on which nobody fell.

        Counts stay the same and nothing is reached; this skips all Agent
        POV work on the common tick without a fall. Returns ``self`` when
        nothing was reached on this frame.
        """
        if not self.oracle_reached and not any(self.agent_pov_reached):
            return self
        return ChampionTakedownsV1(
            oracle_counts=self.oracle_counts,
            agent_pov_counts=self.agent_pov_counts,
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
    "NO_TAKEDOWNS",
    "STREAK_EVERY",
    "ZERO_COUNTS",
    "ChampionStreakV1",
    "ChampionTakedownsV1",
    "TickTakedownsV1",
    "advance_takedown_counts_v1",
    "agent_pov_tick_takedowns_v1",
    "champion_streaks_v1",
    "oracle_tick_takedowns_v1",
]
