"""Decode versioned evaluation feature rows for offline presentation.

The column constants describe the published 58-column wire schema independently
of live Core imports. CONTEXT_FEATURE_* name the context columns that scenes
read; CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH_V2 is column 19, which exists only in
the 20-column context of frame V3. decode_agent_feature_row_v1 reads historical
physical-team identity; decode_agent_feature_row also handles V2 relation flags
with separately authorized display ownership. Version 3 (frame V3, which only
adds a context column) reads agent rows exactly as V2 does. A new wire meaning
needs an explicit version change. A Capture the Flag source (frame V4, context
V5) gives meaning to objective rows 0-3, the two flag homes and the two flags;
decode_capture_the_flag_objective_rows_v1 reads them as the C1.C contract and
decision 139 define them.

Decoders preserve the fields needed for authorized presentation and check exact
Python wire values. They do not infer visibility, hidden state, status sources
or configuration. Results are frozen scalar/tuple records; no renderer, device
execution or file I/O is started by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final

from marl_battlegrounds.evaluation._replays.wire_shapes import SELF_FEATURES_V1

AGENT_FEATURE_X_V1: Final = 0
AGENT_FEATURE_Y_V1: Final = 1
AGENT_FEATURE_RADIUS_V1: Final = 2
AGENT_FEATURE_TEAM_ID_V1: Final = 3
AGENT_FEATURE_ACTIVE_V1: Final = 4
AGENT_FEATURE_ALIVE_V1: Final = 5
AGENT_FEATURE_CLASS_ID_V1: Final = 6
AGENT_FEATURE_BASE_MOVEMENT_SPEED_V1: Final = 7
AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1: Final = 8
AGENT_FEATURE_OBSERVATION_RADIUS_V1: Final = 9
AGENT_FEATURE_BASIC_INTERACTION_RADIUS_V1: Final = 10
AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS_V1: Final = 11
AGENT_FEATURE_CURRENT_HEALTH_V1: Final = 12
AGENT_FEATURE_MAX_HEALTH_V1: Final = 13
AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1: Final = 14

AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION_V1: Final = 15
AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION_V1: Final = 16
AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION_V1: Final = 17
AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1: Final = 18
AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER_V1: Final = 19
AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER_V1: Final = 20
AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1: Final = 21
AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION_V1: Final = 22
AGENT_FEATURE_STUN_ROGUE_POISON_DURATION_V1: Final = 23
AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION_V1: Final = 24
AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1: Final = 25
AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1: Final = 26
AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1: Final = 27
AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1: Final = 28
AGENT_STATUS_FEATURE_START_V1: Final = 15
AGENT_STATUS_FEATURE_STOP_V1: Final = 29

AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1: Final = 29
AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1: Final = 30
AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1: Final = 31

AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE_V1: Final = 32
AGENT_FEATURE_CAPABILITY_BASIC_HEALING_V1: Final = 33
AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION_V1: Final = 34
AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION_V1: Final = 35
AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION_V1: Final = 36
AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION_V1: Final = 37
AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1: Final = 38
AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER_V1: Final = 39
AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER_V1: Final = 40
AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION_V1: Final = 41
AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION_V1: Final = 42
AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION_V1: Final = 43
AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION_V1: Final = 44
AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1: Final = 45
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1: Final = 46
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER_V1: Final = 47
AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1: Final = 48
AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1: Final = 49
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS_V1: Final = 50
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1: Final = 51
AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS_V1: Final = 52
AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1: Final = 53
AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING_V1: Final = 54
AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE_V1: Final = 55
AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS_V1: Final = 56
AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP_V1: Final = 57

# Scientific status channels are not contiguous in presentation order.  These
# tuples name the exact V1 wire columns without importing simulator constants.
AGENT_STATUS_REMAINING_DURATION_COLUMN_BY_CHANNEL_V1: Final = (
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION_V1,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION_V1,
    AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION_V1,
    AGENT_FEATURE_STUN_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1,
)
AGENT_STATUS_ACTIVE_MAGNITUDE_COLUMN_BY_CHANNEL_V1: Final = (
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER_V1,
    AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER_V1,
    None,
    None,
    None,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1,
    None,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1,
)
AGENT_STATUS_CAPABILITY_DURATION_COLUMN_BY_CHANNEL_V1: Final = (
    AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1,
)
AGENT_STATUS_CAPABILITY_MAGNITUDE_COLUMN_BY_CHANNEL_V1: Final = (
    AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER_V1,
    None,
    None,
    None,
    AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1,
)

CONTEXT_FEATURE_MAP_WIDTH_V1: Final = 2
CONTEXT_FEATURE_MAP_HEIGHT_V1: Final = 3
# Frame V3 (context V4) appends the Team Deathmatch Red Zone depth, in map units,
# as context column 19; 19-column contexts (frames V1 and V2) have no such column.
CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH_V2: Final = 19
OBSTACLE_FEATURE_TYPE_V1: Final = 0
OBSTACLE_FEATURE_X_V1: Final = 1
OBSTACLE_FEATURE_Y_V1: Final = 2
OBSTACLE_FEATURE_RADIUS_V1: Final = 3
OBSTACLE_FEATURE_WIDTH_V1: Final = 4
OBSTACLE_FEATURE_HEIGHT_V1: Final = 5
OBSTACLE_FEATURE_THETA_V1: Final = 6
OBSTACLE_FEATURE_ACTIVE_V1: Final = 7
# Capture the Flag objective rows (frame V4, context V5), in the actor's frame:
# its own flag's home and the enemy's, then its own flag and the enemy's.
OBJECTIVE_SLOT_ALLY_FLAG_HOME_V1: Final = 0
OBJECTIVE_SLOT_ENEMY_FLAG_HOME_V1: Final = 1
OBJECTIVE_SLOT_ALLY_FLAG_V1: Final = 2
OBJECTIVE_SLOT_ENEMY_FLAG_V1: Final = 3
OBJECTIVE_FEATURE_ACTIVE_V1: Final = 0
OBJECTIVE_FEATURE_X_V1: Final = 1
OBJECTIVE_FEATURE_Y_V1: Final = 2
# A home row's reach is the capture distance; a flag row's is the pickup and
# return distance. Both are in map units.
OBJECTIVE_FEATURE_INTERACTION_DISTANCE_V1: Final = 3
OBJECTIVE_FEATURE_CONTROLLED_BY_ALLY_TEAM_V1: Final = 4
OBJECTIVE_FEATURE_CONTROLLED_BY_ENEMY_TEAM_V1: Final = 5
OBJECTIVE_FEATURE_STEPS_UNTIL_DROPPED_FLAG_RETURNS_V1: Final = 6
# Columns 7-11: one-hot over the carrier team's five observation rows, set only
# when the actor sees the carrier or is it (decision 155).
OBJECTIVE_FEATURE_CARRIER_ROW_START_V1: Final = 7
OBJECTIVE_FEATURE_CARRIER_ROW_STOP_V1: Final = 12


def _wire_bool(value: float, *, name: str) -> bool:
    """Decode an exact floating-point Boolean from a recorded feature row.

    value must be the Python float 0.0 or 1.0. Return False or True respectively;
    every other type/value raises ValueError labelled with name. No coercion is
    performed, so an integer zero is not accepted as a wire float.
    """
    if type(value) is not float or value not in (0.0, 1.0):
        raise ValueError(f"{name} must be the exact wire float 0.0 or 1.0.")
    return value == 1.0


def _wire_int(value: float, *, name: str, minimum: int = 0) -> int:
    """Decode a finite integer-valued wire float with a lower bound.

    value must be a Python float with no fractional part. Return its Python int
    value when at least minimum, which defaults to zero. ValueError uses name
    for wrong storage, nonfinite/fractional values or a failed lower bound.
    There is no upper-bound or int32-range check here.
    """
    if type(value) is not float or not isfinite(value) or not value.is_integer():
        raise ValueError(f"{name} must be an integer-valued finite wire float.")
    decoded = int(value)
    if decoded < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return decoded


def _finite(value: float, *, name: str, minimum: float | None = None) -> float:
    """Return an unchanged finite Python float after an optional lower-bound check.

    name labels ValueError. minimum defaults to None, meaning no lower bound;
    when supplied it is inclusive. This does not cast other numeric types or
    enforce an upper limit.
    """
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite Python float.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodedAgentFeatureRowV1:
    """Hold the presentation fields decoded from one recorded 58-column agent row.

    All fields are required keyword arguments. Values are Python scalars and
    tuples, not JAX arrays. The frozen, slotted record uses the same field names
    for V1 physical-team rows and V2 relation rows with supplied display ownership.
    Duration and magnitude tuples follow the module's nine-channel wire mapping.

    Notes
    -----
    This constructor only stores values and performs no validation. The decode
    functions check wire types and the bounds described by their contract.
    None in a magnitude tuple means the wire schema has no magnitude column
    for that status; it is different from a stored numeric zero.
    """

    position: tuple[float, float]
    """World [x, y] center as two finite Python floats."""
    radius: float
    """Nonnegative finite body radius in world units."""
    team_id: int
    """Recorded V1 team ID or separately authorized V2 display team.

    The V1 decoder accepts nonnegative integer IDs without checking a live team
    catalog; the V2 caller requires 1 or 2.
    """
    configured_active: bool
    """Whether the recorded row describes a configured participant."""
    alive: bool
    """The recorded current alive flag, decoded from exact zero or one."""
    class_id: int
    """Nonnegative recorded integer class ID, without a live catalog lookup."""
    base_movement_speed: float
    """Nonnegative recorded class speed before current status effects."""
    effective_movement_speed: float
    """Nonnegative recorded movement distance per step in world units."""
    observation_radius: float
    """Nonnegative recorded observation radius in world units."""
    basic_interaction_radius: float
    """Nonnegative recorded Basic interaction radius in world units."""
    ultimate_interaction_radius: float
    """Nonnegative recorded Ultimate interaction radius in world units."""
    current_health: float
    """Recorded nonnegative health, at most maximum_health."""
    maximum_health: float
    """Recorded nonnegative maximum health in health units."""
    ultimate_cooldown_remaining: int
    """Recorded nonnegative integer steps before Ultimate is ready."""
    status_remaining_duration_by_channel: tuple[int, ...]
    """Nine nonnegative remaining duration counts in wire channel order."""
    status_active_magnitude_by_channel: tuple[float | None, ...]
    """Nine finite current magnitudes or None for absent schema columns.

    None does not mean an inactive status; it means the wire has no separate
    magnitude field for that channel.
    """
    steps_until_out_of_combat: int
    """Recorded nonnegative integer recovery countdown in steps."""
    mage_aura_damage_multiplier: float
    """Recorded nonnegative combined Mage outgoing-damage factor."""
    warrior_aura_damage_multiplier: float
    """Recorded nonnegative combined Warrior incoming-damage factor."""
    basic_raw_damage: float
    """Recorded nonnegative Basic damage capability in health units."""
    basic_raw_healing: float
    """Recorded nonnegative Basic healing capability in health units."""
    ultimate_cooldown_steps: int
    """Recorded nonnegative integer Ultimate cooldown capability."""
    status_capability_duration_by_channel: tuple[int, ...]
    """Nine nonnegative catalog-duration counts in wire channel order."""
    status_capability_magnitude_by_channel: tuple[float | None, ...]
    """Nine finite catalog magnitudes or None for absent schema columns."""
    mage_aura_radius: float
    """Recorded nonnegative Mage aura radius in world units."""
    mage_aura_per_emitter_multiplier: float
    """Recorded nonnegative outgoing-damage factor per Mage emitter."""
    warrior_aura_radius: float
    """Recorded nonnegative Warrior aura radius in world units."""
    warrior_aura_per_emitter_multiplier: float
    """Recorded nonnegative incoming-damage factor per Warrior emitter."""
    ultimate_raw_healing: float
    """Recorded nonnegative Ultimate healing capability in health units."""
    ultimate_raw_damage: float
    """Recorded nonnegative Ultimate damage capability in health units."""
    out_of_combat_delay_steps: int
    """Recorded nonnegative integer delay used for combat recovery."""
    out_of_combat_health_regeneration_fraction_per_step: float
    """Recorded nonnegative maximum-health fraction recovered per step.

    The wire decoder checks finiteness and the lower bound, not an upper
    fraction bound or agreement with the live catalog.
    """


def decode_agent_feature_row_v1(row: tuple[float, ...]) -> DecodedAgentFeatureRowV1:
    """Decode one historical row whose team ID is stored in feature column three.

    Parameters
    ----------
    row : tuple of float
        Exact tuple of 58 finite Python floats in the published V1 order.
        Boolean fields must be 0.0/1.0; counters and IDs must be integer-valued.

    Returns
    -------
    DecodedAgentFeatureRowV1
        Immutable scalar/tuple presentation fields. The physical team ID comes
        from the row, with no live Core lookup or inferred identity.

    Raises
    ------
    ValueError
        Tuple/type/length, finite-value, Boolean, integer or nonnegative-field
        checks fail, or current health exceeds maximum health.

    Notes
    -----
    This is a host decoder. It does not check visibility, permissions, class
    catalog agreement, duration maxima or whether the row is a valid Core state.
    It does not mutate data or write a file.
    """
    return _decode_agent_feature_row(row, team_id=None)


def decode_agent_feature_row(
    row: tuple[float, ...], *, schema_version: int, team_id: int, is_enemy: bool
) -> DecodedAgentFeatureRowV1:
    """Decode a recorded agent row using its version and authorized display team.

    Parameters
    ----------
    row : tuple of float
        Exact 58-value tuple of finite Python floats. In V1 column three stores
        team ID; in V2 and V3 it stores the exact zero/one enemy-relation flag.
    schema_version : int
        Version 1, 2 or 3. Version 1 delegates to the historical decoder.
        Version 3 has exactly V2's row meaning (its frames only add context
        column 19, the Red Zone depth, which is not part of an agent row).
    team_id : int
        Authorized display owner, 1 Team A or 2 Team B, required for V2 and V3.
        V1 ignores this argument and uses the recorded physical team ID.
    is_enemy : bool
        Expected V2/V3 observation relation. Column three must equal
        float(is_enemy). V1 ignores this argument.

    Returns
    -------
    DecodedAgentFeatureRowV1
        Common immutable presentation fields. For V2 and V3, team ownership comes
        from team_id rather than treating the enemy flag as a physical identity.

    Raises
    ------
    ValueError
        The version/display team is unsupported, the V2/V3 relation flag
        differs, or an underlying row type, value, length or health check fails.
    TypeError
        Malformed V2/V3 input cannot support the preliminary length/index check
        or conversion of is_enemy to float.

    Notes
    -----
    The caller owns authorization for the supplied display identity. This host
    decoder does not read Oracle state, derive visibility or reconstruct hidden
    simulator facts. No input is changed.
    """
    if schema_version == 1:
        return decode_agent_feature_row_v1(row)
    if (
        type(schema_version) is not int
        or schema_version not in (2, 3)
        or type(team_id) is not int
        or team_id not in (1, 2)
    ):
        raise ValueError("agent row requires a supported schema and display team")
    if len(row) != SELF_FEATURES_V1 or row[3] != float(is_enemy):
        raise ValueError("agent is_enemy flag must match its observation relation")
    return _decode_agent_feature_row(row, team_id=team_id)


def _decode_agent_feature_row(
    row: tuple[float, ...], *, team_id: int | None
) -> DecodedAgentFeatureRowV1:
    """Decode common scalar fields without consulting simulator or Oracle state.

    row must be an exact tuple of 58 finite Python floats. team_id=None reads
    the historical nonnegative integer team column; a supplied team_id replaces
    it and is validated by the versioned caller. Return DecodedAgentFeatureRowV1.
    Check exact wire booleans, nonnegative integer counters/IDs, nonnegative
    radii/health/capabilities and current health at most maximum health.
    Status magnitude columns need only be finite. Catalog maxima, magnitude
    ranges and identity/visibility consistency are not checked. ValueError names
    a failed wire check. Inputs are read only and no arrays are allocated on GPU.
    """
    if type(row) is not tuple or len(row) != SELF_FEATURES_V1:
        raise ValueError(
            f"agent feature row must be an exact {SELF_FEATURES_V1}-value tuple."
        )
    for column, value in enumerate(row):
        _finite(value, name=f"agent feature column {column}")

    position = (
        row[AGENT_FEATURE_X_V1],
        row[AGENT_FEATURE_Y_V1],
    )
    radius = _finite(row[AGENT_FEATURE_RADIUS_V1], name="radius", minimum=0.0)
    maximum_health = _finite(
        row[AGENT_FEATURE_MAX_HEALTH_V1],
        name="maximum health",
        minimum=0.0,
    )
    current_health = _finite(
        row[AGENT_FEATURE_CURRENT_HEALTH_V1],
        name="current health",
        minimum=0.0,
    )
    if current_health > maximum_health:
        raise ValueError("current health must not exceed maximum health.")

    status_remaining = tuple(
        _wire_int(row[column], name=f"status channel {channel} remaining duration")
        for channel, column in enumerate(
            AGENT_STATUS_REMAINING_DURATION_COLUMN_BY_CHANNEL_V1
        )
    )
    status_active_magnitude = tuple(
        None
        if column is None
        else _finite(
            row[column],
            name=f"status channel {channel} active magnitude",
        )
        for channel, column in enumerate(
            AGENT_STATUS_ACTIVE_MAGNITUDE_COLUMN_BY_CHANNEL_V1
        )
    )
    status_capability_duration = tuple(
        _wire_int(row[column], name=f"status channel {channel} capability duration")
        for channel, column in enumerate(
            AGENT_STATUS_CAPABILITY_DURATION_COLUMN_BY_CHANNEL_V1
        )
    )
    status_capability_magnitude = tuple(
        None
        if column is None
        else _finite(
            row[column],
            name=f"status channel {channel} capability magnitude",
        )
        for channel, column in enumerate(
            AGENT_STATUS_CAPABILITY_MAGNITUDE_COLUMN_BY_CHANNEL_V1
        )
    )

    return DecodedAgentFeatureRowV1(
        position=position,
        radius=radius,
        team_id=(
            _wire_int(row[AGENT_FEATURE_TEAM_ID_V1], name="team ID", minimum=0)
            if team_id is None
            else team_id
        ),
        configured_active=_wire_bool(
            row[AGENT_FEATURE_ACTIVE_V1],
            name="configured active",
        ),
        alive=_wire_bool(row[AGENT_FEATURE_ALIVE_V1], name="alive"),
        class_id=_wire_int(row[AGENT_FEATURE_CLASS_ID_V1], name="class ID"),
        base_movement_speed=_finite(
            row[AGENT_FEATURE_BASE_MOVEMENT_SPEED_V1],
            name="base movement speed",
            minimum=0.0,
        ),
        effective_movement_speed=_finite(
            row[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1],
            name="effective movement speed",
            minimum=0.0,
        ),
        observation_radius=_finite(
            row[AGENT_FEATURE_OBSERVATION_RADIUS_V1],
            name="observation radius",
            minimum=0.0,
        ),
        basic_interaction_radius=_finite(
            row[AGENT_FEATURE_BASIC_INTERACTION_RADIUS_V1],
            name="basic interaction radius",
            minimum=0.0,
        ),
        ultimate_interaction_radius=_finite(
            row[AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS_V1],
            name="ultimate interaction radius",
            minimum=0.0,
        ),
        current_health=current_health,
        maximum_health=maximum_health,
        ultimate_cooldown_remaining=_wire_int(
            row[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1],
            name="ultimate cooldown remaining",
        ),
        status_remaining_duration_by_channel=status_remaining,
        status_active_magnitude_by_channel=status_active_magnitude,
        steps_until_out_of_combat=_wire_int(
            row[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1],
            name="steps until out of combat",
        ),
        mage_aura_damage_multiplier=_finite(
            row[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1],
            name="Mage aura aggregate multiplier",
            minimum=0.0,
        ),
        warrior_aura_damage_multiplier=_finite(
            row[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1],
            name="Warrior aura aggregate multiplier",
            minimum=0.0,
        ),
        basic_raw_damage=_finite(
            row[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE_V1],
            name="basic raw damage capability",
            minimum=0.0,
        ),
        basic_raw_healing=_finite(
            row[AGENT_FEATURE_CAPABILITY_BASIC_HEALING_V1],
            name="basic raw healing capability",
            minimum=0.0,
        ),
        ultimate_cooldown_steps=_wire_int(
            row[AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION_V1],
            name="ultimate cooldown capability",
        ),
        status_capability_duration_by_channel=status_capability_duration,
        status_capability_magnitude_by_channel=status_capability_magnitude,
        mage_aura_radius=_finite(
            row[AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS_V1],
            name="Mage aura radius capability",
            minimum=0.0,
        ),
        mage_aura_per_emitter_multiplier=_finite(
            row[AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1],
            name="Mage aura multiplier capability",
            minimum=0.0,
        ),
        warrior_aura_radius=_finite(
            row[AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS_V1],
            name="Warrior aura radius capability",
            minimum=0.0,
        ),
        warrior_aura_per_emitter_multiplier=_finite(
            row[AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1],
            name="Warrior aura multiplier capability",
            minimum=0.0,
        ),
        ultimate_raw_healing=_finite(
            row[AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING_V1],
            name="ultimate raw healing capability",
            minimum=0.0,
        ),
        ultimate_raw_damage=_finite(
            row[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE_V1],
            name="ultimate raw damage capability",
            minimum=0.0,
        ),
        out_of_combat_delay_steps=_wire_int(
            row[AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS_V1],
            name="out-of-combat delay capability",
        ),
        out_of_combat_health_regeneration_fraction_per_step=_finite(
            row[
                AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP_V1
            ],
            name="out-of-combat regeneration capability",
            minimum=0.0,
        ),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureTheFlagFlagRowV1:
    """Hold one flag as an actor's objective row records it.

    This is a frozen, slotted, keyword-only host record. Construction performs
    no validation; decode_capture_the_flag_objective_rows_v1 builds it.

    Attributes
    ----------
    position : tuple[float, float]
        World (x, y) of the flag now, in map units.
    carried : bool
        Whether an agent carries the flag (decision 139's carried bit).
    countdown_steps : int
        Ticks before a dropped flag goes home by itself; 0 unless dropped.
    carrier_row : int | None
        The carrier's row 0-4 among its team's observation rows (the actor's
        ally rows for the enemy flag, its enemy rows for its own flag), or None
        when the row names no carrier: not carried, or carried by an agent the
        actor does not see.
    """

    position: tuple[float, float]
    """World (x, y) of the flag now, in map units."""
    carried: bool
    """Whether an agent carries the flag (decision 139's carried bit)."""
    countdown_steps: int
    """Ticks before a dropped flag goes home by itself; 0 unless dropped."""
    carrier_row: int | None
    """The carrier's row 0-4 among its team's observation rows, or None."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureTheFlagObjectiveRowsV1:
    """Hold an actor's four Capture the Flag objective rows, decoded.

    This is a frozen, slotted, keyword-only host record. Construction performs
    no validation; decode_capture_the_flag_objective_rows_v1 builds it.

    Attributes
    ----------
    ally_home : tuple[float, float]
        World (x, y) of the actor's own flag's home, in map units.
    enemy_home : tuple[float, float]
        World (x, y) of the enemy flag's home, in map units.
    capture_distance : float
        Positive reach of a home for a capture, in map units.
    interaction_distance : float
        Positive reach for a pickup or a return, in map units.
    ally_flag : CaptureTheFlagFlagRowV1
        The actor's own flag.
    enemy_flag : CaptureTheFlagFlagRowV1
        The enemy team's flag.
    """

    ally_home: tuple[float, float]
    """World (x, y) of the actor's own flag's home, in map units."""
    enemy_home: tuple[float, float]
    """World (x, y) of the enemy flag's home, in map units."""
    capture_distance: float
    """Positive reach of a home for a capture, in map units."""
    interaction_distance: float
    """Positive reach for a pickup or a return, in map units."""
    ally_flag: CaptureTheFlagFlagRowV1
    """The actor's own flag."""
    enemy_flag: CaptureTheFlagFlagRowV1
    """The enemy team's flag."""


def decode_capture_the_flag_objective_rows_v1(
    objective_features: tuple[tuple[float, ...], ...],
) -> CaptureTheFlagObjectiveRowsV1:
    """Decode an actor's flag homes and flags from its objective rows.

    Parameters
    ----------
    objective_features : tuple[tuple[float, ...], ...]
        The actor's recorded objective rows, shape (8, 12), as exact Python
        wire floats. Rows 0-3 are read; the hill rows are not.

    Returns
    -------
    CaptureTheFlagObjectiveRowsV1
        Both homes, the two reaches and both flags.

    Raises
    ------
    ValueError
        A flag row is inactive (Core fills these rows from his C1.4 on, so an
        earlier Capture the Flag recording has none); a value is not an exact
        wire float; the two homes or the two flags disagree about their reach,
        or a reach is not positive; a home row names the wrong owner (columns
        4-5 are the actor's own home 1, 0 and the enemy home 0, 1); or a flag's
        bits break decision 139: its own flag is home (1, 0), carried by an
        enemy (0, 1) or dropped (0, 0), and the enemy flag home (0, 1), carried
        by a teammate (1, 0) or dropped (0, 0). A dropped flag counts down from
        at least 1, a home or carried flag does not count, a home flag sits at
        its home, and only a carried flag names one carrier row.

    Notes
    -----
    Every value is copied as recorded; nothing is recomputed from the rules.
    """
    rows = objective_features
    for slot in range(OBJECTIVE_SLOT_ENEMY_FLAG_V1 + 1):
        if not _wire_bool(rows[slot][OBJECTIVE_FEATURE_ACTIVE_V1], name="active"):
            raise ValueError(
                "this Capture the Flag recording has no flag rows; Core fills "
                "them from C1.4 on."
            )

    def point(slot: int) -> tuple[float, float]:
        """Return the finite world (x, y) of objective row slot."""
        return (
            _finite(rows[slot][OBJECTIVE_FEATURE_X_V1], name="objective x"),
            _finite(rows[slot][OBJECTIVE_FEATURE_Y_V1], name="objective y"),
        )

    def reach(slot: int) -> float:
        """Return objective row slot's positive reach, in map units."""
        value = _finite(
            rows[slot][OBJECTIVE_FEATURE_INTERACTION_DISTANCE_V1], name="reach"
        )
        if value <= 0.0:
            raise ValueError("a flag or home reach must be positive.")
        return value

    def bits(slot: int) -> tuple[bool, bool]:
        """Return objective row slot's controlled-by-ally and -enemy bits."""
        return (
            _wire_bool(
                rows[slot][OBJECTIVE_FEATURE_CONTROLLED_BY_ALLY_TEAM_V1],
                name="controlled by ally team",
            ),
            _wire_bool(
                rows[slot][OBJECTIVE_FEATURE_CONTROLLED_BY_ENEMY_TEAM_V1],
                name="controlled by enemy team",
            ),
        )

    def flag(slot: int, home: tuple[float, float]) -> CaptureTheFlagFlagRowV1:
        """Decode flag row slot against its home and decision 139's bits."""
        ally_bit, enemy_bit = bits(slot)
        own_flag = slot == OBJECTIVE_SLOT_ALLY_FLAG_V1
        carried, at_home = (enemy_bit, ally_bit) if own_flag else (ally_bit, enemy_bit)
        countdown = _wire_int(
            rows[slot][OBJECTIVE_FEATURE_STEPS_UNTIL_DROPPED_FLAG_RETURNS_V1],
            name="flag countdown",
        )
        carrier_bits = tuple(
            _wire_bool(value, name="carrier row")
            for value in rows[slot][
                OBJECTIVE_FEATURE_CARRIER_ROW_START_V1:OBJECTIVE_FEATURE_CARRIER_ROW_STOP_V1
            ]
        )
        position = point(slot)
        if (
            (carried and at_home)
            or (countdown > 0) != (not carried and not at_home)
            or (at_home and position != home)
            or sum(carrier_bits) > 1
            or (any(carrier_bits) and not carried)
        ):
            raise ValueError("a flag row breaks decision 139's flag states.")
        return CaptureTheFlagFlagRowV1(
            position=position,
            carried=carried,
            countdown_steps=countdown,
            carrier_row=carrier_bits.index(True) if any(carrier_bits) else None,
        )

    if (
        bits(OBJECTIVE_SLOT_ALLY_FLAG_HOME_V1),
        bits(OBJECTIVE_SLOT_ENEMY_FLAG_HOME_V1),
    ) != ((True, False), (False, True)):
        raise ValueError(
            "a flag home row names its owner: your home 1, 0, the enemy home 0, 1."
        )
    capture_distance = reach(OBJECTIVE_SLOT_ALLY_FLAG_HOME_V1)
    interaction_distance = reach(OBJECTIVE_SLOT_ALLY_FLAG_V1)
    if (
        reach(OBJECTIVE_SLOT_ENEMY_FLAG_HOME_V1) != capture_distance
        or reach(OBJECTIVE_SLOT_ENEMY_FLAG_V1) != interaction_distance
    ):
        raise ValueError("both homes and both flags must share their reaches.")
    ally_home = point(OBJECTIVE_SLOT_ALLY_FLAG_HOME_V1)
    enemy_home = point(OBJECTIVE_SLOT_ENEMY_FLAG_HOME_V1)
    return CaptureTheFlagObjectiveRowsV1(
        ally_home=ally_home,
        enemy_home=enemy_home,
        capture_distance=capture_distance,
        interaction_distance=interaction_distance,
        ally_flag=flag(OBJECTIVE_SLOT_ALLY_FLAG_V1, ally_home),
        enemy_flag=flag(OBJECTIVE_SLOT_ENEMY_FLAG_V1, enemy_home),
    )


__all__ = [
    "AGENT_FEATURE_ACTIVE_V1",
    "AGENT_FEATURE_ALIVE_V1",
    "AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1",
    "AGENT_FEATURE_BASE_MOVEMENT_SPEED_V1",
    "AGENT_FEATURE_BASIC_INTERACTION_RADIUS_V1",
    "AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE_V1",
    "AGENT_FEATURE_CAPABILITY_BASIC_HEALING_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS_V1",
    "AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS_V1",
    "AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE_V1",
    "AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING_V1",
    "AGENT_FEATURE_CLASS_ID_V1",
    "AGENT_FEATURE_CURRENT_HEALTH_V1",
    "AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1",
    "AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1",
    "AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1",
    "AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1",
    "AGENT_FEATURE_MAX_HEALTH_V1",
    "AGENT_FEATURE_OBSERVATION_RADIUS_V1",
    "AGENT_FEATURE_RADIUS_V1",
    "AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1",
    "AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1",
    "AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION_V1",
    "AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER_V1",
    "AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER_V1",
    "AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION_V1",
    "AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1",
    "AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1",
    "AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION_V1",
    "AGENT_FEATURE_STUN_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1",
    "AGENT_FEATURE_TEAM_ID_V1",
    "AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1",
    "AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS_V1",
    "AGENT_FEATURE_X_V1",
    "AGENT_FEATURE_Y_V1",
    "AGENT_STATUS_ACTIVE_MAGNITUDE_COLUMN_BY_CHANNEL_V1",
    "AGENT_STATUS_CAPABILITY_DURATION_COLUMN_BY_CHANNEL_V1",
    "AGENT_STATUS_CAPABILITY_MAGNITUDE_COLUMN_BY_CHANNEL_V1",
    "AGENT_STATUS_FEATURE_START_V1",
    "AGENT_STATUS_FEATURE_STOP_V1",
    "AGENT_STATUS_REMAINING_DURATION_COLUMN_BY_CHANNEL_V1",
    "CONTEXT_FEATURE_MAP_HEIGHT_V1",
    "CONTEXT_FEATURE_MAP_WIDTH_V1",
    "CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH_V2",
    "OBSTACLE_FEATURE_ACTIVE_V1",
    "OBSTACLE_FEATURE_HEIGHT_V1",
    "OBSTACLE_FEATURE_RADIUS_V1",
    "OBSTACLE_FEATURE_THETA_V1",
    "OBSTACLE_FEATURE_TYPE_V1",
    "OBSTACLE_FEATURE_WIDTH_V1",
    "OBSTACLE_FEATURE_X_V1",
    "OBSTACLE_FEATURE_Y_V1",
    "DecodedAgentFeatureRowV1",
    "decode_agent_feature_row",
    "decode_agent_feature_row_v1",
]
