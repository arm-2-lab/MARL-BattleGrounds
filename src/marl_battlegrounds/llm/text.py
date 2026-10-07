"""Render one permitted actor input as compact, exact, readable game text.

format_actor_view is a host-only presentation helper. It never reads Core state,
configuration, another actor's private input or a provider. Its caller chooses
and transforms the coordinate frame before calling it. Every unit is named by
its roster slot (ally_0 ... ally_4, enemy_0 ... enemy_4), the same names the
combat choices use, and the actor is marked "(you)". Each unit's facts appear
once; its own sight and each teammate source keep separate visibility and
permission.
"""

from typing import Literal, cast

import jax
import numpy as np
import numpy.typing as npt

from marl_battlegrounds.core import types as core_types
from marl_battlegrounds.core.combat import GLOBAL_SLOW_FLOOR
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.llm.actions import MOVE_NAMES, TARGET_NAMES, legal_action_names
from marl_battlegrounds.policies.input import ActorInput

# actor-text-v2 (Oct 2026) names units by roster slot instead of v1's per-view
# unit_N IDs, which models misread as slot numbers. Saved runs record it.
TEXT_VERSION = "actor-text-v2"
_DEFAULT_REPLY_INSTRUCTION = (
    'Reply with exactly {"move":"<legal move>","combat":"<legal combat>"}.'
)
type _HostArray = npt.NDArray[np.generic]


def _column_names(prefix: str, size: int) -> tuple[str, ...]:
    """Read ordered field names from Core constants; fail on a changed schema."""
    names = {
        value: name.removeprefix(prefix).lower()
        for name, value in vars(core_types).items()
        if name.startswith(prefix) and isinstance(value, int)
    }
    if set(names) != set(range(size)):
        raise RuntimeError(f"Core's {prefix} schema needs a text-version update")
    return tuple(names[index] for index in range(size))


_UNIT_FIELDS = _column_names("AGENT_FEATURE_", core_types.UNIT_FEATURES)
_CONTEXT_FIELDS = _column_names("CONTEXT_FEATURE_", core_types.CONTEXT_FEATURES)
_OBSTACLE_FIELDS = _column_names("OBSTACLE_FEATURE_", core_types.OBSTACLE_FEATURES)
_RULES = (
    "Rules: Choose one move and one complete combat pair from the legal lists. "
    "All actors commit both choices from the same decision. Damage and healing "
    "are summed before health clipping and death. Combat reads starting positions and "
    "statuses; Charge relocation precedes the chosen ordinary movement. New "
    "statuses first control the next decision. Legal choices can have no effect "
    "because of game physics. Previous actions describe accepted choices, not hits.\n"
    "Abilities: Basic and Ultimate are alternatives. Priest Basic heals an ally, "
    "including self, and grants a temporary speed floor. Other Basics damage an "
    "enemy; Hunter Basic also slows it. Mage Ultimate gives self Burst damage "
    "amplification without a target. Warrior Ultimate damages, slows and stuns "
    "an enemy and moves the "
    "Warrior toward it. Hunter Ultimate damages and traps an enemy. Rogue "
    "Ultimate damages and applies slow, stun and reduced healing. Priest "
    "Ultimate heals an ally. capability_* fields describe what a unit can apply; "
    "other status fields describe effects currently on it. Use the displayed "
    "amounts, ranges and durations. Ultimate starts its full cooldown.\n"
    "Sight and control: Direct sight needs a living observer and target, range "
    "and a clear line through static obstacles; units do not block sight. Shared "
    "sight does not expand your legal combat choices. Shielded enemies are "
    "hidden; shielded units cannot use or receive targeted combat. Stun prevents "
    "movement and combat. Different slows multiply with minimum speed fraction "
    f"{GLOBAL_SLOW_FLOOR}; "
    "Priest's speed floor can raise this. Existing effects age once per tick; "
    "reapplication refreshes durations. Positive damage breaks an existing Hunter "
    "Trap in the next state, but does not remove a Trap first applied that tick. "
    "Poison reduces healing and recovery; it does not damage each tick.\n"
    "Team effects: Mage auras increase outgoing damage; Warrior auras reduce "
    "incoming damage. Living unshielded allies, including the emitter, benefit "
    "within aura range without a sight requirement. Matching auras multiply up "
    "to two emitters' strength. Recovery needs a countdown already at zero and "
    "no combat participation that tick; healing a unit already in combat counts "
    "as participation.\n"
    "Units: Positions, radii, widths and heights are map units; movement speeds "
    "are map units per tick; angles are radians; health, damage and healing use "
    "health units. Durations, cooldowns and clocks count ticks. Multipliers and "
    "fractions are dimensionless. North increases y; east increases x. Diagonal "
    "directions have unit length too. Sight and interaction radii use center-to-"
    "center distance, including their boundary.\n"
    f"Geometry: Obstacle type {core_types.OBSTACLE_TYPE_PILLAR} is a circular pillar "
    f"and {core_types.OBSTACLE_TYPE_WALL} is a rectangular wall. x/y give the center; "
    "radius describes a pillar, and width/height are full wall lengths. theta "
    "rotates a wall counterclockwise from the x axis. Only active obstacles "
    "participate. Bodies have their displayed radius; collisions and map "
    "boundaries can shorten or redirect movement.\n"
    "Roster: ally_0 to ally_4 are your team's five slots, including you; enemy_0 "
    "to enemy_4 are the opponent's. Names never change during a game. A unit you "
    "do not see yourself can still have a line when a teammate sees it. Class IDs are "
    f"{core_types.MAGE_CLASS_ID}=Mage, {core_types.WARRIOR_CLASS_ID}=Warrior, "
    f"{core_types.HUNTER_CLASS_ID}=Hunter, {core_types.ROGUE_CLASS_ID}=Rogue, "
    f"{core_types.PRIEST_CLASS_ID}=Priest; 0 means unused. "
    "Public alive flags reveal no hidden position. Source permission does not "
    "mean the source currently sees a unit. Respawn clock zero means a wave is "
    "due at transition end; a dead actor cannot act before it respawns.\n"
    "Scoring: In Team Deathmatch a death gives the opponent "
    f"{core_types.TEAM_DEATHMATCH_POINTS_PER_DEATH} point, or "
    f"{core_types.TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH} in the victim's own "
    "Red Zone, using its starting position. The zone is the full-height strip "
    "beside that team's spawn edge, including its boundary. Depth is in context; "
    "zero disables it. "
    "A team wins only when it reaches the threshold with more points than the "
    "opponent. A simultaneous threshold tie, or the horizon without a threshold "
    "winner, is a draw.\n"
    "Encoding: Unlisted numeric fields and padded rows are positive zero. "
    "Zero means a whole positive-zero row. Each unit's facts appear once, under "
    "its roster name; visibility and source assignments remain separate. Current "
    "objectives have "
    "8 rows of 12 columns; unexpected nonzero reserved values are printed. "
    "Hidden means no direct visible unit row; Not seen means no previous "
    "accepted action was supplied. All numbers preserve their original values."
)


def _number(value: np.generic) -> str:
    """Spell a finite scalar exactly, retaining float32 precision and signed zero."""
    if value.dtype.kind == "b":
        return "1" if bool(value) else "0"
    if value.dtype.kind in "iu":
        return str(int(value))
    if value.dtype != np.dtype(np.float32) or not np.isfinite(value):
        raise ValueError("Actor text requires finite float32 feature values")
    return min(
        np.format_float_positional(cast(np.float32, value), unique=True, trim="-"),
        np.format_float_scientific(
            cast(np.float32, value), unique=True, trim="-", exp_digits=1
        ),
        key=len,
    )


def _nonzero(value: _HostArray) -> _HostArray:
    """Locate values that cannot be represented by the positive-zero default."""
    return (value != 0) | np.signbit(value)


def _fields(row: _HostArray, names: tuple[str, ...]) -> str:
    """Print one fixed-size float32 row, omitting only positive-zero fields."""
    return (
        " ".join(
            f"{names[index]}={_number(row[index])}"
            for index in np.flatnonzero(_nonzero(row))
        )
        or "Zero"
    )


def _array_text(value: _HostArray) -> str:
    """Print a small host scalar or array with exact leaves and bracketed axes."""
    if value.ndim == 0:
        return _number(cast(np.generic, value[()]))
    return "[" + ",".join(_array_text(row) for row in value) + "]"


def _host_actor(actor: ActorInput) -> ActorInput:
    """Transfer and validate one actor's complete shapes, dtypes and finite facts.

    The caller must supply an already permitted ActorInput. This checks the data
    layout and that self_ally_index names one of the five own-team slots (0-4),
    not information rights, actor identity or decision freshness. Batched inputs
    fail instead of accidentally exposing another actor's private row.
    """
    if not isinstance(actor, ActorInput):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("Input must be one permitted ActorInput")
    host = jax.device_get(actor)
    observation = host.observation
    expected: list[tuple[object, tuple[int, ...], str]] = [
        (observation.self_features, (58,), "float32"),
        (observation.ally_unit_features, (5, 58), "float32"),
        (observation.enemy_unit_features, (5, 58), "float32"),
        (observation.map_obstacle_features, (32, 8), "float32"),
        (observation.objective_features, (8, 12), "float32"),
        (observation.context_features, (20,), "float32"),
        (observation.ally_visibility_mask, (5,), "bool"),
        (observation.enemy_visibility_mask, (5,), "bool"),
        (observation.self_ally_index, (), "int32"),
        (host.source_availability, (5,), "bool"),
        (
            host.source_bank.unit_features_by_source_and_candidate,
            (5, 10, 58),
            "float32",
        ),
        (host.source_bank.unit_visibility_by_source_and_candidate, (5, 10), "bool"),
        (host.source_bank.objective_features_by_source, (5, 8, 12), "float32"),
    ]
    expected.extend(
        (value, (5, size), "float32")
        for value, size in zip(
            observation.previous_timestep_actions, (9, 9, 11, 11, 2, 2), strict=True
        )
    )
    expected.extend(
        (value, shape, dtype)
        for value, (shape, dtype) in zip(
            observation.spawn_lifecycle,
            (
                ((2, 5, 2), "float32"),
                ((2, 5), "int32"),
                ((), "int32"),
                ((), "float32"),
                ((2,), "int32"),
                ((2,), "int32"),
                ((2, 5), "bool"),
                ((2, 5), "bool"),
                ((2, 5), "int32"),
            ),
            strict=True,
        )
    )
    for value, shape, dtype in expected:
        array = np.asarray(value)
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(
                f"Actor field must have shape {shape} and dtype {dtype}; "
                f"got {array.shape}, {array.dtype}"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError("Actor features must be finite")
    if not 0 <= int(np.asarray(observation.self_ally_index)) < 5:
        raise ValueError("self_ally_index must be 0 to 4")
    return host


def _previous(row: _HostArray, names: tuple[str, ...]) -> str:
    """Name a valid one-hot previous action; preserve any other supplied values."""
    indices = np.flatnonzero(row)
    if not np.any(_nonzero(row)):
        return "Not seen"
    if len(indices) == 1 and row[indices[0]] == 1 and not np.any(np.signbit(row)):
        return names[int(indices[0])]
    return "Values" + _array_text(row)


def _objectives(value: _HostArray) -> str:
    """Print nonzero reserved objective rows with their exact row/column positions."""
    return (
        "; ".join(
            f"{index}:{_array_text(row)}"
            for index, row in enumerate(value)
            if np.any(_nonzero(row))
        )
        or "Zero"
    )


_UNIT_NAMES = TARGET_NAMES[1:]
"""Roster names of the ten unit slots: ally_0 ... ally_4, then enemy_0 ... enemy_4."""

_CLASS_NAMES = {
    core_types.MAGE_CLASS_ID: "Mage",
    core_types.WARRIOR_CLASS_ID: "Warrior",
    core_types.HUNTER_CLASS_ID: "Hunter",
    core_types.ROGUE_CLASS_ID: "Rogue",
    core_types.PRIEST_CLASS_ID: "Priest",
}
"""Class names by Core class ID, for the You are and Teams lines; 0 is unused."""


def _teams_line(lifecycle: core_types.SpawnLifecycleObservation, you: int) -> str:
    """Name every slot's class from the public roster arrays, marking self and state.

    Parameters
    ----------
    lifecycle : SpawnLifecycleObservation
        One actor's public lifecycle facts. Its active, alive and class-ID arrays
        have shape (2, 5): own team, then opponent, in roster slot order.
    you : int
        The actor's own slot, 0-4, already checked by ``_host_actor``.

    Returns
    -------
    str
        ``"Teams: ally_0 Mage, ..., ally_4 Priest (you); enemy_0 Mage, ..."``.
        An inactive slot reads ``unused``; an active slot that is not alive
        gains ``dead`` in its parentheses; an unknown class ID reads
        ``class_id N``. The raw arrays still print in the lifecycle block.
    """
    active = np.asarray(lifecycle.active_mask_by_agent_by_team)
    alive = np.asarray(lifecycle.alive_mask_by_agent_by_team)
    classes = np.asarray(lifecycle.class_ids_by_agent_by_team)
    teams: list[str] = []
    for team, relation in enumerate(("ally", "enemy")):
        entries: list[str] = []
        for slot in range(5):
            flags = ["you"] if team == 0 and slot == you else []
            if active[team, slot]:
                class_id = int(classes[team, slot])
                label = _CLASS_NAMES.get(class_id, f"class_id {class_id}")
                if not alive[team, slot]:
                    flags.append("dead")
            else:
                label = "unused"
            marks = f" ({', '.join(flags)})" if flags else ""
            entries.append(f"{relation}_{slot} {label}{marks}")
        teams.append(", ".join(entries))
    return "Teams: " + "; ".join(teams)


def _unit_lines(host: ActorInput) -> list[str]:
    """Print each unit's facts once under its roster name, then who sees what.

    Parameters
    ----------
    host : ActorInput
        One host actor input already checked by ``_host_actor``.

    Returns
    -------
    list of str
        The units header and one line per distinct (unit, row) pair in roster
        order; the sentence explaining "_via_" lines when one exists; "Your own
        row is all zero." when the self row is all zero; the actor's own sight
        (all ten slots); one line for each of the five own-team sources, the
        actor's own included, and its objectives; and the unlisted-candidates
        line.

    Notes
    -----
    A unit's main line, named by its roster name, comes from its first visible
    nonzero row in the order: the actor's self row (self only; it counts as
    visible), the actor's own sight tables, then sources 0-4; with no visible
    nonzero row, from its first nonzero row. Any other distinct nonzero row of
    the same unit gets ``<unit>_via_your_sight`` or ``<unit>_via_ally_<s>``.
    In real games every view of one unit is the same row, so these extra lines
    appear only for synthetic input. Rows of different units are never merged.
    Every input row, visibility flag and permission can be read back exactly.
    """
    obs = host.observation
    you = int(obs.self_ally_index)
    bank = host.source_bank
    self_row = np.asarray(obs.self_features)
    own_rows = np.concatenate(
        (np.asarray(obs.ally_unit_features), np.asarray(obs.enemy_unit_features))
    )
    own_seen = np.concatenate(
        (np.asarray(obs.ally_visibility_mask), np.asarray(obs.enemy_visibility_mask))
    )
    source_rows = np.asarray(bank.unit_features_by_source_and_candidate)
    source_seen = np.asarray(bank.unit_visibility_by_source_and_candidate)

    def shown(unit: int) -> str:
        """Return a unit's roster name, with ``(you)`` for the actor itself."""
        return _UNIT_NAMES[unit] + (" (you)" if unit == you else "")

    names: list[dict[bytes, str]] = []
    definitions: list[str] = []
    for unit in range(10):
        views: list[tuple[str, _HostArray, bool]] = [
            ("your_sight", own_rows[unit], bool(own_seen[unit]))
        ]
        views.extend(
            (
                f"ally_{source}",
                source_rows[source, unit],
                bool(source_seen[source, unit]),
            )
            for source in range(5)
        )
        if unit == you:
            views.insert(0, ("self_row", self_row, True))
        nonzero = [view for view in views if np.any(_nonzero(view[1]))]
        ordered = [view for view in nonzero if view[2]] + nonzero
        unit_names: dict[bytes, str] = {}
        for index, (view, row, _) in enumerate(ordered):
            key = row.tobytes()
            if key not in unit_names:
                name = shown(unit) if index == 0 else f"{_UNIT_NAMES[unit]}_via_{view}"
                unit_names[key] = name.removesuffix(" (you)")
                definitions.append(f"{name}: {_fields(row, _UNIT_FIELDS)}")
        names.append(unit_names)

    def entry(unit: int, row: _HostArray, visible: bool) -> str:
        """Describe one view of one unit for the actor's own sight list."""
        if not np.any(_nonzero(row)):
            return f"{shown(unit)} {'Visible (Zero)' if visible else 'Hidden'}"
        line = names[unit][row.tobytes()]
        if not visible:
            return f"{shown(unit)} Hidden, has data {line}"
        return f"{shown(unit)} Visible" + (
            "" if line == _UNIT_NAMES[unit] else f" ({line})"
        )

    def seen_by_source(unit: int, row: _HostArray) -> str:
        """Describe one unit a teammate source sees, naming any differing line."""
        if not np.any(_nonzero(row)):
            return f"{shown(unit)} (Zero)"
        line = names[unit][row.tobytes()]
        return shown(unit) + ("" if line == _UNIT_NAMES[unit] else f" ({line})")

    lines = [
        "Units (each unit's facts once, under its roster name; "
        "unlisted fields are zero):",
        *definitions,
    ]
    if any(len(unit_names) > 1 for unit_names in names):
        lines.append(
            "A line named like enemy_2_via_ally_3 shows how that view reports "
            "enemy_2 when it differs from enemy_2's line."
        )
    if not np.any(_nonzero(self_row)):
        lines.append("Your own row is all zero.")
    lines.append(
        "Your own sight: "
        + "; ".join(
            entry(unit, own_rows[unit], bool(own_seen[unit])) for unit in range(10)
        )
    )
    for source in range(5):
        label = f"Source ally_{source}" + (" (you)" if source == you else "")
        allowed = bool(host.source_availability[source])
        sees = [
            seen_by_source(unit, source_rows[source, unit])
            for unit in range(10)
            if source_seen[source, unit]
        ]
        hidden = [
            names[unit][source_rows[source, unit].tobytes()]
            for unit in range(10)
            if not source_seen[source, unit]
            and np.any(_nonzero(source_rows[source, unit]))
        ]
        if source == you and not allowed and not sees and not hidden:
            text = "Unavailable; your own sight is listed above"
        else:
            parts = ["Allowed" if allowed else "Unavailable"]
            if sees:
                parts.append("sees " + ", ".join(sees))
            if hidden:
                parts.append("Hidden, has data " + ", ".join(hidden))
            if len(parts) == 1:
                parts.append("No visible rows")
            text = "; ".join(parts)
        objectives = _objectives(np.asarray(bank.objective_features_by_source[source]))
        lines.extend([f"{label}: {text}", f"{label} objectives: {objectives}"])
    lines.append("Unlisted source candidates: Hidden, no data")
    return lines


_COMBAT_NOTE = (
    "Combat: In a combat name, the ally_k or enemy_k part is the target; the "
    "ability is always your own. For example, enemy_k_basic would use your Basic "
    "on enemy_k, and ally_k_ultimate your Ultimate on ally_k. no_combat means no "
    "target and no Ultimate; ultimate means an "
    "untargeted Ultimate. The target and Ultimate marginal masks are the unions "
    "of the listed complete pairs."
)
"""The note under the legal menu. Its example is a pattern (k), never a real
choice, so it cannot point the model at one target or at a masked pair."""


def format_actor_view(
    actor: ActorInput,
    masks: ActionMask,
    *,
    frame: Literal["left", "world"] = "left",
    include_static: bool = True,
    reply_instruction: str
    | None = 'Reply with exactly {"move":"<legal move>","combat":"<legal combat>"}.',
) -> str:
    """Render one permitted current actor view and its complete legal menu.

    Parameters
    ----------
    actor : ActorInput
        One already-filtered actor: self (58,), unit tables (5, 58), obstacles
        (32, 8), context (20,), objectives (8, 12), previous-action tables,
        public spawn lifecycle, source rows (5, 10, 58) and permissions (5,).
        Features are finite float32; masks are Boolean; lifecycle counters are
        int32; self_ally_index is an int32 scalar from 0 to 4. No leading batch
        or actor axis is accepted.
    masks : ActionMask
        Matching actor masks in the same decision epoch and coordinate frame.
    frame : {"left", "world"}, default="left"
        Label of the supplied coordinates, not a request to transform them.
        For left-spawn input, call mirror_team_view with team_on_right first.
        A subsequent action must be mapped back exactly once with mirror_move.

    include_static : bool, default=True
        Include rules and the public map. False is for earlier views in a
        history block whose current view already supplies the same rules/map.
        All changing fields, previous actions and legal choices remain present.
    reply_instruction : str | None
        Final reply-format instruction. The default asks for named JSON. Pass
        a different instruction for a custom parser, or None to omit it in a
        historical view. This changes text only, never the shared action checks.

    Returns
    -------
    str
        Deterministic plain text in this order: the version and frame line; the
        public rules and map (only with include_static); the context; "You are
        ally_k, the <class>." (the class is left out when that slot is inactive
        or its class ID unknown); a Teams line naming every slot's class, with
        "(you)", "dead" and "unused" marks from the public roster arrays; own
        objectives; one line per unit under its roster name (ally_0 ...
        ally_4, enemy_0 ... enemy_4, the actor marked "(you)"), followed, only
        when they apply, by a sentence explaining "_via_" lines and "Your own
        row is all zero."; the actor's own sight for all ten slots; one line
        for each of the five own-team sources, the actor's own included, with
        its permission, the units it sees and its objectives; "Unlisted source
        candidates: Hidden, no data"; the public spawn lifecycle; previous
        actions; legal moves and combat pairs; a fixed note on combat names;
        and the reply instruction. Short decimals round-trip to the original
        float32 including signed zero. Source permission and visibility are
        never combined. Inputs are unchanged.

    Raises
    ------
    ValueError
        Frame, scalar actor shapes, dtypes, finite values or masks are invalid,
        or self_ally_index is outside 0-4.

    Notes
    -----
    Host-only. Each input tree is transferred with device_get before formatting;
    pass NumPy leaves to reuse an earlier batched host transfer. There is no
    provider, persistent cache, tactical selection or private-state access.
    Information filtering and matching actor/epoch/frame are caller obligations.
    Default full-field text is optional; custom methods may format allowed data
    differently while keeping the same action checks.

    The text version is TEXT_VERSION ("actor-text-v2"). Names are roster slots,
    so they are the same in every view and every earlier turn. Each distinct
    row of a unit is printed once. A unit's main line comes from its first
    visible nonzero row (the self row first, then the actor's own sight, then
    sources 0-4); a different row of the same unit, which only synthetic input
    can produce, gets its own line named like enemy_2_via_ally_3 or
    ally_1_via_your_sight. Every input row, visibility flag, permission and
    previous action can be read back exactly from the text. make_system records
    the text version in the System's identity, so a saved run cannot be resumed
    with a different version.
    """
    if frame not in ("left", "world"):
        raise ValueError("Frame must be left or world")
    host = _host_actor(actor)
    menu = legal_action_names(masks)
    obs = host.observation
    lines = [f"MARL-BGs {TEXT_VERSION}; Frame: {frame}"]
    if include_static:
        lines.extend([_RULES, "Map:"])
        obstacles = np.asarray(obs.map_obstacle_features)
        lines.extend(
            f"Obstacle {index}: {_fields(row, _OBSTACLE_FIELDS)}"
            for index, row in enumerate(obstacles)
            if np.any(_nonzero(row))
        )
    you = int(obs.self_ally_index)
    class_id = int(np.asarray(obs.spawn_lifecycle.class_ids_by_agent_by_team)[0, you])
    active = bool(np.asarray(obs.spawn_lifecycle.active_mask_by_agent_by_team)[0, you])
    role = _CLASS_NAMES.get(class_id) if active else None
    lines.extend(
        [
            f"Context: {_fields(np.asarray(obs.context_features), _CONTEXT_FIELDS)}",
            f"You are ally_{you}" + (f", the {role}." if role else "."),
            _teams_line(obs.spawn_lifecycle, you),
            f"Objectives: {_objectives(np.asarray(obs.objective_features))}",
            *_unit_lines(host),
        ]
    )
    lifecycle_names = (
        "Spawn pads",
        "Shield ticks",
        "Configured shield ticks",
        "Shield speed",
        "Wave periods",
        "Wave countdowns",
        "Active roster",
        "Alive roster",
        "Class IDs",
    )
    lines.append("Spawn lifecycle (team axes: own team, opponent):")
    lines.extend(
        f"{name}: {_array_text(np.asarray(value))}"
        for name, value in zip(lifecycle_names, obs.spawn_lifecycle, strict=True)
    )
    previous = obs.previous_timestep_actions
    lines.append("Previous accepted actions (targets already use your roster names):")
    for relation, moves, targets, ultimates in (
        ("ally", previous[0], previous[2], previous[4]),
        ("enemy", previous[1], previous[3], previous[5]),
    ):
        for index in range(5):
            values = (
                _previous(np.asarray(moves[index]), MOVE_NAMES),
                _previous(np.asarray(targets[index]), TARGET_NAMES),
                _previous(np.asarray(ultimates[index]), ("No", "Yes")),
            )
            mark = " (you)" if relation == "ally" and index == you else ""
            lines.append(
                f"{relation}_{index}{mark}: "
                + (
                    "Not seen"
                    if values == ("Not seen",) * 3
                    else f"move={values[0]} target={values[1]} ultimate={values[2]}"
                )
            )
    lines.extend(
        [
            "Legal moves: " + ", ".join(menu["move"]),
            "Legal combat pairs: " + ", ".join(menu["combat"]),
            _COMBAT_NOTE,
        ]
    )
    if reply_instruction is not None:
        lines.append(reply_instruction)
    return "\n".join(lines)
