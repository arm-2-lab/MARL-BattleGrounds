"""Read the installed Capture the Flag map catalog and verify its packaged maps.

Capture the Flag owns its 52 maps (decision 97): data/maps/ and data/manifest.json
in this folder, copied once from Team Deathmatch's on 7 November 2026 and owned by
Capture the Flag from then on. This module reads only that folder. It reuses Team
Deathmatch's record models (TDMMapInfo, MapGeometry), so an entry has the same
fields in both modes, and keeps its own reads, cache and messages. Each entry's
source names the authored Team Deathmatch draft it was copied from. There is no
map history or alias table yet; they begin with the first change to a Capture the
Flag map. No game, JAX device or training distribution is created here.
"""

from __future__ import annotations

import hashlib
from functools import cache
from importlib.resources import files
from typing import Literal

from marl_battlegrounds.evaluation.models import EvaluationModel
from marl_battlegrounds.modes.team_deathmatch._assets import MapGeometry, TDMMapInfo


class CTFAssetManifest(EvaluationModel):
    """Store the versioned installed Capture the Flag map catalog.

    Attributes
    ----------
    schema_id : str, default="marl_battlegrounds.ctf_assets"
        Fixed format identity.
    schema_version : int, default=1
        Supported format version.
    maps : tuple[TDMMapInfo, ...]
        Map entries, named "ctf_map_id_<id>_<words>_<split>"; asset_manifest
        checks ordered IDs 0 through 51.

    Notes
    -----
    Constructing the model validates its declared fields. Scenarios join with
    Capture the Flag's own scenario content.
    """

    schema_id: Literal["marl_battlegrounds.ctf_assets"] = (
        "marl_battlegrounds.ctf_assets"
    )
    schema_version: Literal[1] = 1
    maps: tuple[TDMMapInfo, ...]


def _resource_bytes(relative_path: str) -> bytes:
    """Read one known resource below the installed modes/capture_the_flag/data folder.

    relative_path is an internal package-relative name. Return raw bytes; ordinary
    resource and file errors propagate. Call it only with owned resource names.
    """
    return (
        files("marl_battlegrounds")
        .joinpath("modes", "capture_the_flag", "data", relative_path)
        .read_bytes()
    )


@cache
def asset_manifest() -> CTFAssetManifest:
    """Load, validate and cache the installed Capture the Flag manifest.

    Returns
    -------
    CTFAssetManifest
        Ordered catalog with maps 0 through 51.

    Raises
    ------
    ValueError
        Ordered map IDs do not cover 0 through 51.

    Notes
    -----
    Package read and Pydantic schema errors also propagate. The first successful
    call reads manifest.json; later calls reuse the cached frozen model. Map
    resource digests are checked when each map is loaded.
    """
    manifest = CTFAssetManifest.model_validate_json(_resource_bytes("manifest.json"))
    if tuple(row.map_id for row in manifest.maps) != tuple(range(52)):
        raise ValueError(
            "Capture the Flag package must contain ordered maps 0 through 51"
        )
    return manifest


def current_map_id(info: TDMMapInfo) -> int:
    """Return a map ID only when all supplied details are the current CTF entry.

    Parameters
    ----------
    info : TDMMapInfo
        Map entry selected from list_ctf_maps.

    Returns
    -------
    int
        The entry's current public ID.

    Raises
    ------
    ValueError
        Its ID is invalid or any field differs from the installed entry, for
        example a Team Deathmatch entry, whose name differs.
    """
    entries = asset_manifest().maps
    if (
        type(info.map_id) is not int
        or not 0 <= info.map_id < len(entries)
        or info != entries[info.map_id]
    ):
        raise ValueError(
            "map details do not match the installed Capture the Flag catalogue; "
            "choose the map again with list_ctf_maps() from marl_battlegrounds.tasks"
        )
    return info.map_id


def map_geometry(info: TDMMapInfo) -> MapGeometry:
    """Read and verify the packaged geometry for a selected Capture the Flag map.

    Parameters
    ----------
    info : TDMMapInfo
        Entry from list_ctf_maps giving the map ID and expected byte digest.

    Returns
    -------
    MapGeometry
        Parsed geometry with the original ordered obstacle rows and spawn banks.

    Raises
    ------
    ValueError
        The file's bytes do not match the entry's digest ("Capture the Flag
        package content digest mismatch: maps/<id>.json"). Schema validation
        can also reject malformed content.

    Notes
    -----
    File errors propagate. This verifies content bytes, not physical validity;
    the config factory runs Core's checks.
    """
    relative = f"maps/{info.map_id}.json"
    payload = _resource_bytes(relative)
    if hashlib.sha256(payload).hexdigest() != info.resource_sha256:
        raise ValueError(
            f"Capture the Flag package content digest mismatch: {relative}"
        )
    return MapGeometry.model_validate_json(payload)
