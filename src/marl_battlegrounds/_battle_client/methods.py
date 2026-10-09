"""Load live-client methods once and compose ordered physical slot assignments.

The launch owns this resolver's cache and resource leases. Existing load_method,
shared_policy and team remain the loading, scalar adaptation and composition
owners. No checkpoint format, method registry or private source search is added.
Manual members produce neutral rows; the live turn owner replaces those rows
with validated human drafts immediately before submission. Each slot's label
is the name of the method that plays it, which the clients show and saved
replays record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from marl_battlegrounds._battle_client.control import (
    _manual_policy,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds._battle_client.model import BUILT_IN_CONTROLLER_NAMES
from marl_battlegrounds._battle_client.protocol import ExecutionInformationMode
from marl_battlegrounds._battle_client.setup import BattleSetup, Controller
from marl_battlegrounds._method_loading import (
    _reference_key,  # pyright: ignore[reportPrivateUsage]
    _reference_kind,  # pyright: ignore[reportPrivateUsage]
    load_method,
)
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    shared_policy,
    team,
)
from marl_battlegrounds.evaluation.recording_identity import team_controller_names

_BUILTIN_NAMES = BUILT_IN_CONTROLLER_NAMES


@dataclass(frozen=True, slots=True)
class LoadedControllers:
    """Keep resolved methods and ownership for one blocking client launch.

    controllers holds the existing bounded Team A/B labels. systems supplies
    their System descriptors, or None for a legacy Manual/fixed opponent route.
    methods contains each distinct loaded/adapted object once for launch leases.
    manual_slots and slot_labels use the ten physical rows, including configured
    dead actors; inactive rows have no label. Each label is the name of the
    method that plays that slot, exactly as given ("Manual" for a human); a
    team(...) System names the member that owns the slot
    (recording_identity.team_controller_names). original_methods retains the exact
    returned objects before Policy adaptation so owned replacement can reject a
    reused factory object. No method or parameter is copied.
    """

    controllers: tuple[str, str]
    systems: tuple[System | None, System | None] | None
    methods: tuple[System, ...]
    manual_slots: tuple[int, ...]
    slot_labels: tuple[str | None, ...]
    original_methods: tuple[Policy | System, ...] = ()


def _is_model_file(path: Path) -> bool:
    """Return whether path is an existing portable model file (a checked ZIP)."""
    if not path.is_file():
        return False
    from marl_battlegrounds.portable_models import is_portable_model_file

    return is_portable_model_file(path)


def load_checkpoint_path(path: str | Path) -> Policy | System:
    """Load a saved model from private browser input.

    Accepted: an existing checkpoint directory, a portable model file, or a
    released model (a typed name such as
    ``MARL-BattleGrounds/tdm-season-0-qmix@final``, an ``hf://`` address or a
    released-layout folder), through the shared load_method grammar. A released
    model is downloaded once into Hugging Face's standard cache when missing.
    Relative paths use the server's working directory. Expansion of ~ is allowed.
    Reject missing paths and other files before calling the shared loader.
    Browser text is never interpreted as a Python factory expression or a
    built-in name. Reject symbolic links in checkpoint folders before resolving
    the path. Native checkpoint checks and optional training dependencies remain
    with the shared loader. Raises ValueError for anything else.
    """
    if isinstance(path, str) and path.strip() and _reference_kind(path) == "released":
        return load_method(_reference_key(path))
    target = Path(path).expanduser().absolute()
    if _is_model_file(target):
        return load_method(_reference_key(str(target)))
    if not target.is_dir():
        raise ValueError(
            "Choose an existing checkpoint folder, a model file or a released "
            "model name."
        )
    return load_method(_reference_key(str(target)))


def validate_controller_information(
    setup: BattleSetup, execution_information_mode: ExecutionInformationMode
) -> tuple[str, str] | None:
    """Check information-mode support before loading any method.

    SharedObs uses the existing System composition and returns None. NoSharedObs
    retains the private client's old whole-team Manual/Random route and returns
    its two legacy controller names. Uniform per-slot lists count as whole-team
    choices. Other NoSharedObs assignments fail before loading; the client does
    not give a method a wider input contract to make it run.
    """
    if execution_information_mode == "shared_obs":
        return None
    if execution_information_mode != "no_shared_obs":
        raise ValueError("Choose SharedObs or NoSharedObs.")
    aliases = {
        "manual": "manual",
        "random": "random_valid",
        "random_valid": "random_valid",
    }
    controllers: list[str] = []
    for source in (setup.team_a, setup.team_b):
        entries = source if isinstance(source, tuple) else (source,)
        if not entries or any(
            not isinstance(value, str) or value not in aliases for value in entries
        ):
            raise ValueError(
                "NoSharedObs supports whole teams using Manual or Random. "
                "Choose SharedObs for other controllers or mixed slots."
            )
        names = {aliases[value] for value in entries if isinstance(value, str)}
        if len(names) != 1:
            raise ValueError(
                "NoSharedObs supports whole teams using Manual or Random. "
                "Choose SharedObs for other controllers or mixed slots."
            )
        controllers.append(names.pop())
    return controllers[0], controllers[1]


def resolve_controllers(
    setup: BattleSetup,
    *,
    execution_information_mode: ExecutionInformationMode = "shared_obs",
) -> LoadedControllers:
    """Load validated sources once and group repeated members through team().

    setup comes from prepare_battle_setup, which checks both rosters and lists
    before any method is loaded. Built-in names and bare module:function strings
    keep the shared loader's priority, even beside same-named folders. Explicit
    Path inputs stay paths. Reject linked paths before cache lookup; ordinary
    equivalent paths share one load. Other strings keep their exact reference.
    Direct objects share only by identity. A distinct object never merges because
    of equality.
    Released model names and portable model files load through the same
    shared loader; a name and its exact address share one load.
    Whole-team Systems retain their identity; Policy adaptation retains the
    original callable and variable leaves. Each slot's label is its method's
    name for that slot (team_controller_names), so a whole-team team(...)
    System lists its members by champion. Initialization, resource entry and
    model calls remain the launch/worker's responsibility. Loader errors propagate.
    execution_information_mode defaults to SharedObs. NoSharedObs preserves only
    the private client's existing whole-team Manual/Random execution path; other
    choices are rejected before loading. The public launcher always uses SharedObs.
    """
    legacy = validate_controller_information(setup, execution_information_mode)
    config, _ = setup.scenario.build_scenario()
    active = np.asarray(config.agent_profile.active_mask, dtype=np.bool_)
    classes = np.asarray(config.agent_profile.class_ids).astype(int).tolist()
    if legacy is not None:
        legacy_manual_slots = tuple(
            slot for slot in range(10) if active[slot] and legacy[slot // 5] == "manual"
        )
        legacy_labels = tuple(
            None
            if not active[slot]
            else "Manual"
            if slot in legacy_manual_slots
            else "Random"
            for slot in range(10)
        )
        return LoadedControllers(legacy, None, (), legacy_manual_slots, legacy_labels)
    cache: dict[tuple[str, str | int], System] = {}
    adapted: dict[int, System] = {}
    manual = shared_policy(Policy("Manual", _manual_policy))
    loaded: list[System] = []
    original_methods: list[Policy | System] = []

    def resolve(source: Controller) -> System:
        """Reuse a source or adapt its exact returned object without freezing it."""
        if isinstance(source, (str, Path)):
            text = str(source)
            if text == "manual" and isinstance(source, str):
                return manual
            if isinstance(source, Path):
                path = source.expanduser().absolute()
                if not path.is_dir() and not _is_model_file(path):
                    raise ValueError(f"Checkpoint folder does not exist: {path}")
                text = str(path)
            else:
                text = _BUILTIN_NAMES.get(text, text)
            key: tuple[str, str | int] = ("reference", _reference_key(text))
        else:
            key = ("object", id(source))
        if key in cache:
            return cache[key]
        value = load_method(str(key[1])) if key[0] == "reference" else source
        assert isinstance(value, (System, Policy))
        if id(value) not in adapted:
            original_methods.append(value)
            adapted[id(value)] = (
                value if isinstance(value, System) else shared_policy(value)
            )
            loaded.append(adapted[id(value)])
        cache[key] = adapted[id(value)]
        return cache[key]

    names: list[str] = []
    systems: list[System | None] = []
    manual_slots: list[int] = []
    labels: list[str | None] = [None] * 10
    for index, source in enumerate((setup.team_a, setup.team_b)):
        physical = tuple(
            slot for slot in range(index * 5, index * 5 + 5) if active[slot]
        )
        entries = source if isinstance(source, tuple) else (source,) * len(physical)
        if len(entries) != len(physical):
            raise ValueError(
                "Controller lists must match every configured roster slot."
            )
        if index == 1 and setup.scenario.scenario_specification is not None:
            # Preserve the scenario opponent owner and its recorded legacy identity.
            assert isinstance(source, str) and source in _BUILTIN_NAMES
            names.append(source)
            systems.append(None)
            for slot in physical:
                labels[slot] = _BUILTIN_NAMES[source]
            continue
        if all(isinstance(entry, str) and entry == "manual" for entry in entries):
            names.append("manual")
            systems.append(None)
            manual_slots.extend(physical)
            for slot in physical:
                labels[slot] = "Manual"
            continue
        members: list[System] = []
        owned: list[list[int]] = []
        for slot, entry in zip(physical, entries, strict=True):
            method = resolve(entry)
            # A team(...) entry names the member that owns this slot.
            labels[slot] = team_controller_names(
                method, classes[index * 5 : index * 5 + 5]
            )[slot % 5]
            if method is manual:
                manual_slots.append(slot)
            position = next(
                (i for i, member in enumerate(members) if member is method), None
            )
            if position is None:
                position = len(members)
                members.append(method)
                owned.append([])
            owned[position].append(slot % 5)
        names.append(f"system:team_{'a' if index == 0 else 'b'}")
        systems.append(members[0] if len(members) == 1 else team(*members, slots=owned))
    pair = None if all(value is None for value in systems) else (systems[0], systems[1])
    return LoadedControllers(
        (names[0], names[1]),
        pair,
        tuple(loaded),
        tuple(manual_slots),
        tuple(labels),
        tuple(original_methods),
    )
