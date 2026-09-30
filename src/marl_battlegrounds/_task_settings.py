"""Own host task names, canonical rosters and the default Red Zone depth.

Configuration checks use these names without constructing numerical class IDs.
The tasks module owns class-ID conversion and re-exports the existing names.
"""

from collections.abc import Sequence
from typing import Literal, cast, get_args

type AgentClassName = Literal["mage", "warrior", "hunter", "rogue", "priest"]


DEFAULT_TDM_RED_ZONE_DEPTH = 5.0


_CLASS_NAMES = get_args(AgentClassName.__value__)
_CANONICAL_ROSTER: tuple[AgentClassName, ...] = cast(
    tuple[AgentClassName, ...], _CLASS_NAMES
)


def _validate_roster(  # pyright: ignore[reportUnusedFunction] - Shared host check.
    roster: Sequence[AgentClassName], *, name: str
) -> None:
    """Check one ordered team roster without building class-ID arrays.

    Accept one through five supported class names, including repeats. name is
    the team label used in errors. Reject strings and unordered containers with
    TypeError, and unsupported names or team sizes with ValueError. No I/O.
    """
    if isinstance(roster, (str, bytes)) or not isinstance(
        cast(object, roster), Sequence
    ):
        raise TypeError(f"{name} must be an ordered sequence of class names")
    if not 1 <= len(roster) <= 5:
        raise ValueError(f"{name} must contain one through five agents")
    for value in roster:
        if type(value) is not str or value not in _CLASS_NAMES:
            raise ValueError(f"{name} has unsupported class {value!r}")


def canonical_tournament_rosters() -> tuple[
    tuple[AgentClassName, ...], tuple[AgentClassName, ...]
]:
    """Return the default ordered five-agent roster for each team.

    Takes no arguments and performs no device work or configuration validation.
    Factories use this same order when the caller does not supply rosters.

    Returns
    -------
    tuple[tuple[AgentClassName, ...], tuple[AgentClassName, ...]]
        Two immutable tuples, for Team A and Team B. Each contains ``mage``,
        ``warrior``, ``hunter``, ``rogue`` and ``priest`` in that order. Order
        assigns the first five team slots and their matching ordered spawn pads.
        The tuples are shared constants and can be reused safely.
    """
    return _CANONICAL_ROSTER, _CANONICAL_ROSTER
