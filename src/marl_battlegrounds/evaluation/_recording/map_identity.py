"""Read and verify recorded map names without changing simulator conditions.

Host-only helpers match explicit asset identity or registered metadata against
the current catalog, retained map history and approved aliases. They also own
each map's readable name, which the clients show; the technical name stays the
stored identity. New map evidence (registered_map_metadata) and the readers
of saved records (recorded_map) both use the record's own mode's catalog: Team
Deathmatch's, with its history and aliases, or Capture the Flag's own maps.
Geometry alone never grants an old recording a map ID or split. Custom layouts
remain unnamed unless the record provides truthful descriptive text. No JAX
runtime is imported here.
"""

from __future__ import annotations

import re
from array import array
from collections.abc import Mapping
from functools import lru_cache
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import Field

from marl_battlegrounds._task_settings import (
    SPAWN_EXCHANGE_FIELDS,
    TASK_MODES,
    TaskName,
    task_for_mode,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    EvaluationEpisodeContext,
    EvaluationModel,
    ResolvedEnvConfigCaptureTheFlagV3,
    ResolvedEnvConfigV1,
    ResolvedEnvConfigV2,
)
from marl_battlegrounds.modes.capture_the_flag import _assets as ctf_assets
from marl_battlegrounds.modes.team_deathmatch._assets import (
    MapGeometry,
    TDMMapInfo,
    asset_manifest,
    map_geometry,
    map_history,
    map_id_aliases,
)


class RecordedMap(EvaluationModel):
    """Validated display identity for one recorded layout.

    Attributes
    ----------
    map_id : Annotated[int, Field(ge=0, le=51)] | None
        Registered integer ID 0..51, or None when no approved identity is known.
    technical_name : Annotated[str, Field(min_length=1)]
        Nonempty stable source name or recorded layout identifier.
    display_name : Annotated[str, Field(min_length=1)]
        Nonempty readable label. A registered map shows its technical name's
        words in title case ("Teutoburg Forest") unless it has its own
        readable name ("Jamyla's Grove" for map 50). Custom maps use
        "Custom Map". The label is built when the record is read, never stored.
    split : Literal['training', 'validation', 'test'] | None
        "training", "validation", "test", or None when not established.

    This frozen host model is descriptive metadata, not a simulator config.
    """

    map_id: Annotated[int, Field(ge=0, le=51)] | None
    technical_name: Annotated[str, Field(min_length=1)]
    display_name: Annotated[str, Field(min_length=1)]
    split: Literal["training", "validation", "test"] | None


def _float32(values: tuple[float, ...]) -> tuple[float, ...]:
    """Round Python values to the float32 values stored in recorded geometry."""
    return tuple(array("f", values))


def approved_map_id(asset_id: str | None, semantic_digest: str) -> int | None:
    """Find a current map whose explicit authored-source identity matches.

    Parameters
    ----------
    asset_id : str | None
        Declared source asset name, or None.
    semantic_digest : str
        Declared digest of that source's semantic content.

    Returns
    -------
    int | None
        Current integer map ID, or None when no source name/digest pair matches.

    Notes
    -----
        This cached-catalog host lookup does not infer identity from geometry or
        validate a source revision number. It does not edit map metadata.
    """
    return next(
        (
            row.map_id
            for row in asset_manifest().maps
            if row.source.asset_id == asset_id
            and row.source.semantic_digest == semantic_digest
        ),
        None,
    )


# Readable names that differ from the words of a map's technical name, keyed by
# those words joined with hyphens. Only the label people see changes: the
# technical name stays the map's stored identity in results, replays, file names
# and training records. The key also matches older names of the same map under
# its historical IDs (for example "tdm-map-id-7-nzoth-the-corruptor-training").
_DISPLAY_NAMES: dict[str, str] = {
    "pyramids-of-giza": "Pyramids of Giza",
    "nimbus-class-v-wing": "Nimbus-class V-wing",
    "x-wing-starfighter": "X-wing Starfighter",
    "nzoth-the-corruptor": "Whispers of N'Zoth",
    "crows-nest": "Crow's Nest",
    "stairway-to-heaven": "Stairway to Heaven",
    "blades-edge": "Blade's Edge",
    "ufo": "UFO",
    "wall-e": "WALL-E",
    "three-body-problem": "Three-Body Problem",
    "jamylas-grove": "Jamyla's Grove",
}


def _map_identity(
    map_id: int, technical_name: str, split: Literal["training", "validation", "test"]
) -> RecordedMap:
    """Build a readable label from a verified ID, technical name and split.

    The label is the technical name without its "tdm-map-id-N-" or
    "ctf_map_id_N_" prefix and split suffix, in title case ("Teutoburg
    Forest"), unless _DISPLAY_NAMES
    gives that map another name ("Jamyla's Grove"). The technical name is
    returned unchanged.
    """
    prefix = r"^(?:tdm|ctf)[-_]map[-_]id[-_]\d+[-_]"
    name = re.sub(prefix, "", technical_name)
    name = re.sub(rf"[-_]{split}$", "", name)
    words = re.sub(r"[-_]+", "-", name)
    display = _DISPLAY_NAMES.get(words) or words.replace("-", " ").title()
    return RecordedMap(
        map_id=map_id,
        technical_name=technical_name,
        display_name=display,
        split=split,
    )


@lru_cache(maxsize=104)
def _registered_map(map_id: int, task: TaskName) -> RecordedMap:
    """Look up a current map ID in one task's catalog; reject IDs outside it.

    task "tdm" reads Team Deathmatch's manifest and "ctf" Capture the Flag's
    own. An invalid ID raises ValueError ("recorded map_id must identify an
    approved TDM map", or "... CTF map").
    """
    entries = (asset_manifest() if task == "tdm" else ctf_assets.asset_manifest()).maps
    if type(map_id) is not int or not 0 <= map_id < len(entries):
        raise ValueError(
            f"recorded map_id must identify an approved {task.upper()} map"
        )
    info = entries[map_id]
    return _map_identity(map_id, info.name, info.split)


@lru_cache(maxsize=104)
def _registered_map_by_name(
    name: str, task: TaskName = "tdm"
) -> tuple[RecordedMap, int]:
    """Resolve an exact current or approved historical technical name.

    Return the recorded identity and its catalog-family ID separately. task
    picks the catalog: "tdm" (the default) also accepts Team Deathmatch's
    approved historical names and aliases, unchanged; "ctf" accepts only
    Capture the Flag's current names, since that catalog has no history yet.
    Unknown names raise ValueError.
    """
    if task == "ctf":
        for info in ctf_assets.asset_manifest().maps:
            if info.name == name:
                return _registered_map(info.map_id, "ctf"), info.map_id
        raise ValueError("recorded map name conflicts with the approved catalogue")
    for info in asset_manifest().maps:
        if info.name == name:
            return _registered_map(info.map_id, "tdm"), info.map_id
    for row in map_history():
        if row.info.name == name:
            info = row.info
            return _map_identity(info.map_id, info.name, info.split), info.map_id
    for alias in map_id_aliases().maps:
        if alias.name == name:
            historical = map_history()[alias.current_map_id].info
            return _map_identity(
                alias.map_id, alias.name, historical.split
            ), alias.current_map_id
    raise ValueError("recorded map name conflicts with the approved catalogue")


@lru_cache(maxsize=156)
def _authored_map(
    asset_id: str, revision: int, digest: str, task: TaskName = "tdm"
) -> tuple[RecordedMap, MapGeometry] | None:
    """Match an explicit asset name, revision and digest to current or approved old
    sources. Return its recorded name and that exact revision's geometry, or None.
    task picks the catalog: Team Deathmatch checks its current catalog first and
    loads history only when it does not match; Capture the Flag has only its
    current catalog, with no history yet.
    """
    if task == "ctf":
        for info in ctf_assets.asset_manifest().maps:
            source = info.source
            if (
                source.asset_id == asset_id
                and source.revision == revision
                and source.semantic_digest == digest
            ):
                return _registered_map(info.map_id, "ctf"), ctf_assets.map_geometry(
                    info
                )
        return None
    map_id = approved_map_id(asset_id, digest)
    if map_id is not None and revision == asset_manifest().maps[map_id].source.revision:
        return _registered_map(map_id, "tdm"), map_geometry(
            asset_manifest().maps[map_id]
        )
    for row in map_history():
        source = row.info.source
        if (
            source.asset_id == asset_id
            and source.revision == revision
            and source.semantic_digest == digest
        ):
            info = row.info
            return _map_identity(info.map_id, info.name, info.split), row.geometry
    for alias in map_id_aliases().maps:
        if (
            alias.asset_id == asset_id
            and alias.revision == revision
            and alias.semantic_digest == digest
        ):
            row = map_history()[alias.current_map_id]
            return (
                _map_identity(alias.map_id, alias.name, row.info.split),
                row.geometry,
            )
    return None


class MapLayout(NamedTuple):
    """A map's mode and geometry, for checking a config that has no resolved record.

    Attributes
    ----------
    task_mode : int
        The config's EnvConfig.task_mode, which picks the map registry (Team
        Deathmatch's or Capture the Flag's), as a resolved record's does.
    map_width, map_height : float
        World-space map dimensions.
    obstacles : tuple[tuple[float, ...], ...]
        Ordered obstacle rows: type, x, y, radius, width, height, angle in
        radians and activity, each as its float32 value.
    team_spawn_pad_positions : tuple[tuple[tuple[float, ...], ...], ...]
        Team A then Team B's ordered (x, y) spawn banks, as float32 values.

    Notes
    -----
    It holds no rules: a map's identity is its mode's registry entry and its
    geometry. Flag homes are not map geometry. evaluation_conditions builds it
    from an EnvConfig. Immutable and hashable, so registered_map_metadata can
    cache it; two modes' layouts of one map differ in task_mode, so they never
    share a cached answer.
    """

    task_mode: int
    map_width: float
    map_height: float
    obstacles: tuple[tuple[float, ...], ...]
    team_spawn_pad_positions: tuple[tuple[tuple[float, ...], ...], ...]


def _geometry_matches(
    config: ResolvedEnvConfigV1
    | ResolvedEnvConfigV2
    | ResolvedEnvConfigCaptureTheFlagV3
    | MapLayout,
    geometry: MapGeometry,
    *,
    mode: int,
) -> bool:
    """Compare a recorded config or a layout with one registry's packaged geometry.

    mode is the registry's Core mode (TASK_MODES: 1 for Team Deathmatch, 3 for
    Capture the Flag). The record's or layout's task_mode must equal it, so a
    Team Deathmatch-only path refuses a Capture the Flag layout. Then check
    dimensions, every ordered float32 obstacle row and both ordered spawn banks.
    Roster and other rules do not change map identity. Neither immutable input
    is changed; this does not simulate or read files.
    """
    if config.task_mode != mode:
        return False
    if not isinstance(config, MapLayout):
        config = MapLayout(
            config.task_mode,
            config.map_width,
            config.map_height,
            tuple(
                (
                    float(row.obstacle_type_id),
                    row.x,
                    row.y,
                    row.radius,
                    row.width,
                    row.height,
                    row.theta,
                    float(row.is_active),
                )
                for row in config.obstacle_slots
            ),
            config.team_spawn_pad_positions,
        )
    return config == MapLayout(
        mode,
        geometry.map_width,
        geometry.map_height,
        tuple(_float32(row) for row in geometry.obstacles),
        tuple(
            tuple(_float32(point) for point in team)
            for team in geometry.team_spawn_pad_positions
        ),
    )


@lru_cache(maxsize=128)
def registered_map_metadata(
    map_id: int,
    config: ResolvedEnvConfigV1
    | ResolvedEnvConfigV2
    | ResolvedEnvConfigCaptureTheFlagV3
    | MapLayout,
) -> tuple[AggregationKeyV1, ...]:
    """Verify a declared map's geometry and return stable recording keys.

    Parameters
    ----------
    map_id : int
        Current approved integer map ID in the record's own mode's catalog
        (Team Deathmatch's or Capture the Flag's).
    config : ResolvedEnvConfigV1 | ResolvedEnvConfigV2 |
    ResolvedEnvConfigCaptureTheFlagV3 | MapLayout
        Frozen resolved config for the actual episode (V2 records the Red Zone
        depth, which does not affect map geometry; V3 is Capture the Flag's), or
        the episode's MapLayout, which evaluation uses for configs of any mode.

    Returns
    -------
    tuple[AggregationKeyV1, ...]
        Four AggregationKeyV1 values naming map_id, map_name, map_origin and
        map_split. Successful results are cached for this immutable config.

    Raises
    ------
    ValueError
        The map ID is invalid, the record's mode has no map registry (only Team
        Deathmatch and Capture the Flag have one), or dimensions, ordered
        obstacle slots or ordered spawn banks differ from the mode's asset.

    Notes
    -----
        Host-only. This is exact registered-geometry verification, including spawn
        order; it does not accept an exchanged bank by guessing a source relation.
        Roster and other non-geometry rules do not establish or change map identity.
    """
    # The record's own mode picks the registry. A mode with none (neutral, King
    # of the Hills) is checked against Team Deathmatch's, whose mode it fails.
    task = task_for_mode(config.task_mode) or "tdm"
    identity = _registered_map(map_id, task)
    geometry = (
        map_geometry(asset_manifest().maps[map_id])
        if task == "tdm"
        else ctf_assets.map_geometry(ctf_assets.asset_manifest().maps[map_id])
    )
    if not _geometry_matches(config, geometry, mode=TASK_MODES[task]):
        raise ValueError("recorded map_id does not match the episode geometry")
    return (
        AggregationKeyV1(name="map_id", value=str(map_id)),
        AggregationKeyV1(name="map_name", value=identity.technical_name),
        AggregationKeyV1(name="map_origin", value="registered"),
        AggregationKeyV1(name="map_split", value=str(identity.split)),
    )


def _snapshot_map_metadata(  # pyright: ignore[reportUnusedFunction]
    map_id: int,
    config: ResolvedEnvConfigV1
    | ResolvedEnvConfigV2
    | ResolvedEnvConfigCaptureTheFlagV3
    | MapLayout,
    declared: Mapping[str, Any],
) -> tuple[AggregationKeyV1, ...]:
    """Verify one pinned map revision without replacing it with today's catalog.

    The private tournament bridge supplies serialized TDMMapInfo and the actual
    source configuration (a resolved record or a MapLayout), before any spawn
    exchange. Require an approved current
    or historical source, exact name/ID/split and exact geometry. Return the same
    recording keys as registered_map_metadata. Invalid evidence raises ValueError;
    no files, model calls or simulator work occur here.
    """
    info = TDMMapInfo.model_validate(declared)
    approved = _authored_map(
        info.source.asset_id, info.source.revision, info.source.semantic_digest
    )
    if (
        approved is None
        or info.map_id != map_id
        or approved[0].map_id != map_id
        or approved[0].technical_name != info.name
        or approved[0].split != info.split
        or not _geometry_matches(config, approved[1], mode=TASK_MODES["tdm"])
    ):
        raise ValueError("snapshot map identity does not match its approved geometry")
    return (
        AggregationKeyV1(name="map_id", value=str(map_id)),
        AggregationKeyV1(name="map_name", value=info.name),
        AggregationKeyV1(name="map_origin", value="registered"),
        AggregationKeyV1(name="map_split", value=info.split),
    )


@lru_cache(maxsize=128)
def _recorded_map(
    bindings: tuple[tuple[str, str], ...], layout: str, task: TaskName
) -> RecordedMap:
    """Interpret immutable recorded keys without inventing historical identity.

    Explicit registered keys must contain a coherent ID, name and split, read
    in task's catalog (the record's own mode). Custom keys cannot also claim
    registered metadata. Older free-form names remain descriptive only.
    Invalid explicit claims raise ValueError.
    """
    metadata = dict(bindings)
    origin = metadata.get("map_origin")
    if origin == "custom":
        if metadata != {"map_origin": "custom"}:
            raise ValueError("custom map metadata cannot claim a registered identity")
        return RecordedMap(
            map_id=None,
            technical_name=layout,
            display_name="Custom Map",
            split=None,
        )
    if origin == "registered":
        if set(metadata) != {"map_origin", "map_id", "map_name", "map_split"}:
            raise ValueError("recorded map identity requires ID, name and split")
        try:
            map_id = int(metadata["map_id"])
        except ValueError as error:
            raise ValueError("recorded map_id must be an integer") from error
        result, _ = _registered_map_by_name(metadata["map_name"], task)
        if map_id != result.map_id or metadata["map_split"] != result.split:
            raise ValueError("recorded map name or split conflicts with its map_id")
        return result
    if origin is not None:
        raise ValueError("recorded map_origin must be registered or custom")
    # Historical aggregation keys were free-form. A recorded name is useful,
    # but neither its text nor arbitrary old IDs establish an approved split.
    name = metadata.get("map_name")
    return RecordedMap(
        map_id=None,
        technical_name=name or layout,
        display_name=name or "Map name unavailable",
        split=None,
    )


def _declared_source_geometry(
    context: EvaluationEpisodeContext,
) -> ResolvedEnvConfigV1 | ResolvedEnvConfigV2 | ResolvedEnvConfigCaptureTheFlagV3:
    """Return source-order geometry only for an explicit verified spawn marker.

    Older records without the namespaced marker keep their exact bank order.
    New markers require a source-content ID and choice 0 or 1. Choice 1 reverses
    the team axis of every recorded SPAWN_EXCHANGE_FIELDS field (both complete
    banks, and the flag homes where the record has them) for identity checking
    only; the replay's actual config stays unchanged. This checks a declared map
    relationship, not an unspecified geometry match. Writer creation separately
    verifies full source content.
    """
    names = {"marl_bgs.spawn_locations", "marl_bgs.source_config_id"}
    rows = tuple(row for row in context.aggregation_keys if row.name in names)
    if not rows:
        return context.resolved_env_config
    bindings = {row.name: row.value for row in rows}
    if len(rows) != 2 or set(bindings) != names:
        raise ValueError(
            "recorded spawn locations require one source ID and one choice"
        )
    if re.fullmatch(r"[0-9a-f]{64}", bindings["marl_bgs.source_config_id"]) is None:
        raise ValueError("recorded spawn source ID must be a SHA-256 content identity")
    choice = bindings["marl_bgs.spawn_locations"]
    if choice not in ("0", "1"):
        raise ValueError("recorded spawn locations must be 0 or 1")
    resolved = context.resolved_env_config
    if choice == "0":
        return resolved
    return resolved.model_copy(
        update={
            name: tuple(reversed(getattr(resolved, name)))
            for name in SPAWN_EXCHANGE_FIELDS
            if name in type(resolved).model_fields
        }
    )


def recorded_map(context: EvaluationEpisodeContext) -> RecordedMap:
    """Return a verified display identity for an episode's recorded layout.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Immutable evaluation context containing layout identity,
        aggregation keys and the exact resolved episode config.

    Returns
    -------
    RecordedMap
        RecordedMap preserving a verified current or approved historical name.
        Without such evidence, return a custom/descriptive identity with no
        registered map ID or split.

    Raises
    ------
    ValueError
        Explicit map keys conflict, a declared name is unknown, or
        the episode geometry disagrees with its map revision, or an explicit
        authored source conflicts with the registered metadata.

    Notes
    -----
        Host-only and read-only. An exact authored name/revision/digest may identify
        an older recording without keys. An explicit verified spawn choice permits
        checking the complete exchanged banks against that source. Records without
        that marker still need an exact bank match. Matching geometry is never used
        to invent a historical map assignment.
    """
    source_geometry = _declared_source_geometry(context)
    bindings = tuple(
        (row.name, row.value)
        for row in context.aggregation_keys
        if row.name in {"map_id", "map_name", "map_split", "map_origin"}
    )
    layout = context.identity.layout
    # The record's own mode picks the catalog, as on the writing side.
    task = task_for_mode(context.resolved_env_config.task_mode) or "tdm"
    result = _recorded_map(bindings, layout.identifier, task)
    if not bindings:
        # Older authored recordings already name and fingerprint their source
        # layout. Recognize that explicit identity, never geometry alone.
        authored = _authored_map(
            layout.identifier, layout.version, layout.canonical_digest
        )
        if authored is not None:
            result, geometry = authored
            if not _geometry_matches(source_geometry, geometry, mode=TASK_MODES["tdm"]):
                raise ValueError("recorded source does not match the episode geometry")
        return result
    if result.map_id is None:
        return result
    if layout.identifier != "resolved-layout":
        # An explicit source selects one version. A conflicting source cannot
        # borrow another version merely because its geometry would match.
        authored = _authored_map(
            layout.identifier, layout.version, layout.canonical_digest, task
        )
        if authored is None or authored[0] != result:
            raise ValueError("recorded source identity conflicts with its map metadata")
        if not _geometry_matches(source_geometry, authored[1], mode=TASK_MODES[task]):
            raise ValueError("recorded source does not match the episode geometry")
        return result
    _, map_id = _registered_map_by_name(result.technical_name, task)
    if task == "ctf":
        ctf_map = ctf_assets.asset_manifest().maps[map_id]
        if _geometry_matches(
            source_geometry, ctf_assets.map_geometry(ctf_map), mode=TASK_MODES["ctf"]
        ):
            return result
        raise ValueError("recorded map_id does not match the episode geometry")
    current = asset_manifest().maps[map_id]
    if result.technical_name == current.name and _geometry_matches(
        source_geometry, map_geometry(current), mode=TASK_MODES["tdm"]
    ):
        return result
    historical = map_history()[map_id]
    if _geometry_matches(source_geometry, historical.geometry, mode=TASK_MODES["tdm"]):
        return result
    raise ValueError("recorded map_id does not match the episode geometry")
