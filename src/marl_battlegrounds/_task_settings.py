"""Own host task names, rosters, each mode's defaults and the spawn exchange.

Configuration checks use these names without constructing numerical class IDs.
The tasks module owns class-ID conversion and re-exports the existing names.
This module imports no JAX, so record readers can use it too.
"""

from collections.abc import Mapping, Sequence
from typing import Literal, cast, get_args

type AgentClassName = Literal["mage", "warrior", "hunter", "rogue", "priest"]

# Task names that make() accepts, one per mode: "tdm" (Team Deathmatch) and
# "ctf" (Capture the Flag). make, evaluate and training refuse any other value
# through check_task_name below.
type TaskName = Literal["tdm", "ctf"]
TASK_NAMES: tuple[TaskName, ...] = get_args(TaskName.__value__)


# The points a Team Deathmatch team needs to win.
DEFAULT_TDM_SCORE_THRESHOLD = 20

# The one default Team Deathmatch Red Zone depth, in map units. Fresh map
# configs, default evaluation, new training runs and new DevClient drafts use it;
# 0.0 keeps one point per death. Saved records keep the depth they recorded.
DEFAULT_TDM_RED_ZONE_DEPTH = 5.0

# The Capture the Flag defaults: captures needed to win; the largest distance,
# in map units, from an agent's center to a flag for a pickup or a return, and
# from a carrier's center to its own flag's home for a capture; and the ticks a
# dropped flag waits before going home. Each flag home is its team's spawn
# center, the same on every map, Team A's first, in the map's own frame.
DEFAULT_CTF_SCORE_THRESHOLD = 3
DEFAULT_CTF_FLAG_INTERACTION_DISTANCE = 0.75
DEFAULT_CTF_FLAG_CAPTURE_DISTANCE = 1.25
DEFAULT_CTF_DROPPED_FLAG_AUTO_RETURN_STEPS = 30
DEFAULT_CTF_FLAG_HOME_POSITIONS = ((0.5, 5.0), (19.5, 5.0))

# The King of the Hills defaults, used by its test config builder until the
# mode gets a task name: points per tick for each owned hill with no enemy on
# it; extra points per tick while one team owns all three hills that way; the
# empty ticks in a row a hill stays its owner's; each hill's radius in map
# units; the hill centers on the 20 by 10 maps, by slot (slot 0 left of the
# midline, slot 1 on it, slot 2 right of it); and the points needed to win
# with three hills and with one.
DEFAULT_KOTH_HILL_POINTS_PER_STEP = 1
DEFAULT_KOTH_THREE_HILL_SWEEP_BONUS_POINTS_PER_STEP = 3
DEFAULT_KOTH_EMPTY_HILL_OWNERSHIP_STEPS = 0
DEFAULT_KOTH_HILL_RADIUS = 1.25
DEFAULT_KOTH_HILL_CENTERS = ((5.0, 5.0), (10.0, 5.0), (15.0, 5.0))
DEFAULT_KOTH_SCORE_THRESHOLD = 100
DEFAULT_ONE_HILL_KOTH_SCORE_THRESHOLD = 50

# The Core mode each task name plays (EnvConfig.task_mode): Team Deathmatch is
# 1 and Capture the Flag 3, the values of core.types.TASK_MODE_TDM and
# TASK_MODE_CTF, written here as plain integers because Core loads JAX and this
# module must not (map identity reads them). A test pins them to Core's. King of
# the Hills joins with its task name.
TASK_MODES: Mapping[TaskName, int] = {"tdm": 1, "ctf": 3}


def task_for_mode(task_mode: int) -> TaskName | None:
    """Return the task name that plays one Core mode, or None if none does.

    Parameters
    ----------
    task_mode : int
        An EnvConfig.task_mode value, for example 3.

    Returns
    -------
    TaskName | None
        ``"tdm"`` for 1 and ``"ctf"`` for 3. None for the neutral mode 0, King
        of the Hills (2) and any other value. Host-only; imports no JAX.
    """
    return next((name for name, mode in TASK_MODES.items() if mode == task_mode), None)


def check_task_name(task: object) -> TaskName:
    """Return a task name unchanged, or refuse a value that is not one.

    Parameters
    ----------
    task : object
        The value a caller passed as a task name, for example ``"ctf"``.

    Returns
    -------
    TaskName
        ``task`` itself, typed as a task name.

    Raises
    ------
    ValueError
        If ``task`` is not in TASK_NAMES. The message lists the task names, for
        example "task must be one of 'tdm', 'ctf', not 'dm'". For ``"koth"`` it
        also says King of the Hills arrives in v1.2.0: Core plays that mode
        already, but it has no task name, maps or bots yet.

    Notes
    -----
    The one owner of the refusal shared by make, evaluate, the training config,
    the training schedule, validation and the task dispatchers. Host-only;
    imports no JAX.
    """
    if task in TASK_NAMES:
        return task
    message = f"task must be one of {', '.join(map(repr, TASK_NAMES))}, not {task!r}"
    if task == "koth":
        message += " (King of the Hills arrives in v1.2.0)"
    raise ValueError(message)


# Each task's defaults, read by every caller that serves more than one mode:
# the score a team needs to win (TDM points, CTF captures), and the Red Zone
# depth in map units (CTF has no Red Zone, so 0.0). King of the Hills joins
# with its task name.
DEFAULT_SCORE_THRESHOLDS: Mapping[TaskName, int] = {
    "tdm": DEFAULT_TDM_SCORE_THRESHOLD,
    "ctf": DEFAULT_CTF_SCORE_THRESHOLD,
}
DEFAULT_RED_ZONE_DEPTHS: Mapping[TaskName, float] = {
    "tdm": DEFAULT_TDM_RED_ZONE_DEPTH,
    "ctf": 0.0,
}

# The EnvConfig fields whose team axis, each game's first axis, a spawn exchange
# reverses, so each team takes the other's spawn pads and flag home. Every
# exchange (tasks._swap_spawn_banks) and every reader of an exchanged record
# uses this one list. Hills need no entry: which hill is a team's Spawn Side
# Hill follows from its pads.
SPAWN_EXCHANGE_FIELDS = (
    "team_spawn_pad_positions",
    "capture_the_flag_flag_home_positions",
)


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
