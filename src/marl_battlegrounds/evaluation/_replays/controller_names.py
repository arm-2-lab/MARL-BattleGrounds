"""Record and read the controller names a replay keeps.

A replay keeps who played each team and each champion as aggregation keys in
its episode context, so no record schema changes:

- ``marl_bgs.system_name.team_a`` and ``marl_bgs.system_name.team_b``: each
  team's controller name (evaluate has always written these).
- ``marl_bgs.controller_name.slot_0`` to ``slot_9``: the controller of each
  used champion slot (global slot order, 0 to 4 Team A, 5 to 9 Team B), added
  2 October 2026. A team built from several methods (``marl_bgs.team``) names
  each member's champions.

Each value is the name as a JSON string with non-ASCII characters escaped, so
any Unicode name fits the ASCII metadata contract. Names are kept exactly as
given ("tdm-beta" stays "tdm-beta"); they are display text, never identities.

Writers: ``team_controller_name_key`` and ``slot_controller_name_keys``
(recording_context for evaluate, the BattleClient/DevClient evaluation bridge).
Readers: ``recorded_team_controller_name`` and
``recorded_slot_controller_names`` (the Replay Viewer's match summary). Old
replays without the slot keys read as None, and readers fall back to team
names. This module imports no JAX and touches no files.

Agent POV: who controls each champion is fixed before the game and is not
hidden information.
"""

import json
from collections.abc import Sequence
from typing import cast

from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    EvaluationEpisodeContext,
)

TEAM_CONTROLLER_NAME_KEY = "marl_bgs.system_name.{team}"
"""Aggregation key of one team's controller name; team is team_a or team_b."""

SLOT_CONTROLLER_NAME_KEY = "marl_bgs.controller_name.slot_{slot}"
"""Aggregation key of one champion slot's controller name; slot is 0 to 9."""


def _name_value(name: str) -> str:
    """Encode a display name as a JSON string so any Unicode name fits ASCII."""
    return json.dumps(name, ensure_ascii=True)


def _decoded_name(value: str | None) -> str | None:
    """Decode one recorded JSON name; None when absent, malformed or blank."""
    if value is None:
        return None
    try:
        name = json.loads(value)
    except ValueError:
        return None
    return name if isinstance(name, str) and name.strip() else None


def team_controller_name_key(team: str, name: str) -> AggregationKeyV1:
    """Build the aggregation key that records one team's controller name.

    Parameters
    ----------
    team : {"team_a", "team_b"}
        Which team the name belongs to.
    name : str
        The team's controller name exactly as shown (a System's own name, or
        for the live clients the names of its champions' controllers joined
        with " / "). Any Unicode text; it must not be blank.

    Returns
    -------
    AggregationKeyV1
        Key ``marl_bgs.system_name.team_a`` (or ``team_b``) whose value is the
        name as a JSON string with non-ASCII characters escaped, so it fits
        the ASCII metadata contract. These are the bytes evaluate has always
        written.

    Raises
    ------
    ValueError
        team is not team_a or team_b, or name is blank.
    """
    if team not in ("team_a", "team_b"):
        raise ValueError("team must be team_a or team_b")
    if not name.strip():
        raise ValueError("a recorded controller name must not be blank")
    return AggregationKeyV1(
        name=TEAM_CONTROLLER_NAME_KEY.format(team=team), value=_name_value(name)
    )


def slot_controller_name_keys(
    names: Sequence[str | None],
) -> tuple[AggregationKeyV1, ...]:
    """Build one aggregation key per champion slot that names its controller.

    Parameters
    ----------
    names : sequence of str or None
        Ten entries in global slot order (0 to 4 Team A, 5 to 9 Team B). A
        string is the controller of that slot exactly as shown ("Manual" for
        a human-played slot); None means the slot is unused and gets no key.

    Returns
    -------
    tuple of AggregationKeyV1
        Keys ``marl_bgs.controller_name.slot_0`` to ``slot_9`` for the named
        slots, in slot order. Each value is the name as a JSON string with
        non-ASCII characters escaped. The caller sorts all of a context's keys.

    Raises
    ------
    ValueError
        names does not hold ten entries, or a name is blank or not a string.
    """
    rows = tuple(names)
    if len(rows) != 10:
        raise ValueError("slot controller names need ten entries")
    keys: list[AggregationKeyV1] = []
    for slot, name in enumerate(rows):
        if name is None:
            continue
        if not isinstance(cast(object, name), str) or not name.strip():
            raise ValueError("slot controller names must be nonempty strings")
        keys.append(
            AggregationKeyV1(
                name=SLOT_CONTROLLER_NAME_KEY.format(slot=slot),
                value=_name_value(name),
            )
        )
    return tuple(keys)


def recorded_team_controller_name(
    context: EvaluationEpisodeContext, team: str
) -> str | None:
    """Read one team's recorded controller name from a replay context.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Any recorded episode context (every version has aggregation keys).
    team : {"team_a", "team_b"}
        Which team to read.

    Returns
    -------
    str or None
        The name exactly as recorded, or None when the context has no such
        key (older BattleClient and DevClient replays) or its value is not a
        nonempty JSON string.
    """
    key = TEAM_CONTROLLER_NAME_KEY.format(team=team)
    return _decoded_name(
        next((row.value for row in context.aggregation_keys if row.name == key), None)
    )


def recorded_slot_controller_names(
    context: EvaluationEpisodeContext,
) -> tuple[str | None, ...] | None:
    """Read each champion slot's recorded controller name from a replay context.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Any recorded episode context.

    Returns
    -------
    tuple of str or None, or None
        Ten entries in global slot order: the name exactly as recorded, or
        None for a slot without a key (an unused slot). The whole result is
        None when the context records no slot names at all, as in replays
        saved before 2 October 2026; callers then fall back to team names.
    """
    values = {row.name: row.value for row in context.aggregation_keys}
    names = tuple(
        _decoded_name(values.get(SLOT_CONTROLLER_NAME_KEY.format(slot=slot)))
        for slot in range(10)
    )
    return None if all(name is None for name in names) else names
