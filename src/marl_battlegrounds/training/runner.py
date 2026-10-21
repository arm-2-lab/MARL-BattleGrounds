"""Run and resume one declared PPO, recurrent QMIX or recurrent PQN-VDN experiment.

TrainConfig is re-exported from training.config, the host settings owner.
train joins existing collection,
learner, checkpoint and evaluation authorities; numerical work stays outside
host logging and file handling. CLI commands use these same public functions.
All methods share the run layout, validation, selection and reports; the
saved method chooses the learner, its settings block and its host counts.
A new or child run saves its records only after it holds its folder's run lock
and finds the folder still empty apart from the lock file. Released model names
in the settings and population bindings become exact ``hf://`` addresses before
any run file is written (released_models owns those rules).
"""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from marl_battlegrounds.baselines.config import (
    QMIXConfig,
    pqn_planned_learning_blocks,
)
from marl_battlegrounds.baselines.methods import (
    is_ppo_method,
    method_settings_field,
)
from marl_battlegrounds.training._continuation_schedules import (
    CONTINUATION_CHANGE_KEYS,
)
from marl_battlegrounds.training._run_io import (
    ProgressReporter,
    TrainingSpeedEstimate,
    append_jsonl,
    atomic_json,
    log_cursor,
    preserve_log_suffix,
    process_identity,
    qmix_fixed_block_counts,
    restore_log_cursor,
    run_lock,
    set_aside_abandoned_exports,
    study_attempt_binding,
    unused_run_folder,
    utc_now,
    validate_host_state,
    validate_log_cursor,
)
from marl_battlegrounds.training.config import (
    TrainConfig as TrainConfig,
)
from marl_battlegrounds.training.config import (
    _check_happo_rosters,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.training.config import (
    config_from_dict as config_from_dict,
)
from marl_battlegrounds.training.config import (
    config_to_dict as config_to_dict,
)
from marl_battlegrounds.training.config import (
    read_config as read_config,
)

if TYPE_CHECKING:
    from jax import Array

    from marl_battlegrounds.core.types import EnvState
    from marl_battlegrounds.environment import TrainingFacts
    from marl_battlegrounds.evaluation._recording.recording_checkpoint import (
        PreparedRecordingFork,
    )
    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.training._content import PreparedTrainingContent
    from marl_battlegrounds.training.checkpoints import RestoredCheckpoint
    from marl_battlegrounds.training.collection import (
        TrainingCarry,
        TrainingCollection,
        TrainingRollout,
    )
    from marl_battlegrounds.training.curriculum import TrainingSchedule
    from marl_battlegrounds.training.learner import LearnerState, UpdateResult
    from marl_battlegrounds.training.shaping import RewardOutput
    from marl_battlegrounds.training.validation import FrozenPanel

    type _Updater = Callable[
        [LearnerState, TrainingCarry, TrainingRollout],
        tuple[LearnerState, UpdateResult],
    ]


@dataclass(frozen=True)
class TrainResult:
    """Return saved paths and exact counts after all declared work succeeds.

    run_dir identifies the experiment; final_actor and selected_actor identify
    frozen exports (selection is None without a panel or an eligible trained
    candidate). completed_env_steps and
    completed_updates exclude padding; for QMIX completed_updates counts
    optimizer steps (epochs per learning block) and for PQN-VDN optimizer
    steps (epochs times minibatches per learning block), not blocks. status is
    complete only after validation, export and reporting finish.
    evidence_paths names the shared report outputs. final_checkpoint names the
    complete saved learner that can be resumed or passed to extend_training.
    This record carries no rollout, live model state or learning qualification.
    """

    run_dir: Path
    final_actor: Path
    selected_actor: Path | None
    completed_env_steps: int
    completed_updates: int
    status: Literal["complete"]
    evidence_paths: tuple[Path, ...]
    final_checkpoint: Path


def load_run(run_dir: str | Path) -> TrainResult:
    """Read a complete saved training result without training or writing reports.

    run_dir is the run root, relative to the working directory when needed.
    Return the same path/count record as train, using latest_checkpoint.json and
    saved selected/final identities rather than a filename ranking. New saved
    report paths are relative to the run. Historical runs reconstruct only known
    existing report filenames. Moved local actors use the same basename under
    actors/; an inherited selected actor keeps its external path when needed.

    Raises ValueError/OSError for incomplete/recovering runs, inconsistent saved
    identities/counters, missing actors or reports. Reads small descriptions and
    records only; load_system separately checks and loads numerical actor bytes.
    A complete software run does not establish policy quality.
    """
    from marl_battlegrounds.training._checkpoint_metadata import read_resume_checkpoint

    root = Path(run_dir).expanduser().resolve()
    info = json.loads((root / "run_details.json").read_text())
    status = json.loads((root / "status.json").read_text())
    if not isinstance(info, dict) or not isinstance(status, dict):
        raise ValueError("Saved run and status must be JSON objects")
    info, status = cast(dict[str, Any], info), cast(dict[str, Any], status)
    if (
        status.get("status") != "complete"
        or (root / "checkpoint_recovery.json").exists()
    ):
        raise ValueError(
            f"Training run is not complete: {root} ({status.get('status')})"
        )
    checkpoint, details = read_resume_checkpoint(root)
    from marl_battlegrounds.training._run_io import saved_result_actors

    _, final, selected = saved_result_actors(root, details, status, run_details=info)
    assert final is not None  # Complete runs require a checked final actor.
    recorded = status.get("evidence_paths")
    if recorded is None:
        # These are the historical runner's fixed local reports, never a ranking.
        names = [
            "learning_curves.png",
            "learning_curves.svg",
            "learning_curve.csv",
            "validation_cells.csv",
            "exposure_history.csv",
            "run_summary.md",
        ]
        method = info["config"].get("method", "mappo")
        if method in ("qmix", "pqn_vdn"):
            family = "qmix" if method == "qmix" else "pqn"
            names.extend([f"{family}_curves.png", f"{family}_curves.svg"])
        recorded = [name for name in names if (root / name).is_file()]
    if not isinstance(recorded, list) or not recorded:
        raise ValueError(f"Complete run has no saved report paths: {root}")
    from marl_battlegrounds.training._checkpoint_metadata import (
        _relative,  # pyright: ignore[reportPrivateUsage]
    )

    paths = tuple(_relative(root, name) for name in cast(list[object], recorded))
    if any(not path.is_file() or not path.stat().st_size for path in paths):
        raise ValueError(f"Complete run is missing a saved report: {root}")
    return TrainResult(
        root,
        final,
        selected,
        status["env_steps"],
        status["completed_updates"],
        "complete",
        paths,
        checkpoint,
    )


def _preserve_reports(root: Path, attempt: str) -> None:
    """Archive existing derived reports before moving their active continuation.

    Preserve raw bytes, including malformed reports, under their content hashes
    in the old attempt folder. Fsync before the caller clears active summaries.
    Immutable evaluation tasks remain in their original directories. The caller
    holds the run lock and has validated the complete recovery checkpoint.
    """
    directory = root / "attempts" / attempt / "reports"
    for name in (
        "validation_results.json",
        "random_diagnostics.json",
        "selection.json",
        "exposure.json",
        "exposure_history.csv",
        "slot_diagnostic.json",
        "run_summary.md",
    ):
        source = root / name
        if not source.is_file():
            continue
        raw = source.read_bytes()
        directory.mkdir(parents=True, exist_ok=True)
        destination = (
            directory / f"{source.stem}-{sha256(raw).hexdigest()}{source.suffix}"
        )
        if destination.exists():
            if destination.read_bytes() != raw:
                raise ValueError("Archived report identity conflicts")
            continue
        temporary = directory / f".{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def _restored_reward_shape(
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    progress: Array,
    *,
    component_keys: tuple[str, ...] = (),
) -> Array | RewardOutput:
    """Build the old reward's output shape for an explicitly changed-reward fork.

    The saved identity and numerical state remain the parent's. This pure zero
    callback is used only while constructing its restore template. The child
    installs its checked new reward before any collection or learning call.
    component_keys are the checked parent's saved names; they reconstruct its
    pending component window. Empty names preserve historical array-only state.
    """
    del before, facts, after, progress
    import jax.numpy as jnp

    from marl_battlegrounds.training.shaping import RewardOutput

    value = jnp.zeros((10,), dtype=jnp.float32)
    if component_keys:
        return RewardOutput(value, {name: value for name in component_keys})
    return value


def _history_setup(
    config: TrainConfig,
    schedule: TrainingSchedule,
    prepared: PreparedTrainingContent,
    *,
    check_spacing: bool = True,
) -> tuple[TrainingSchedule, dict[str, int]]:
    """Resolve the rolling bank once and reject impossible publication spacing.

    Reuse the learner's collection-boundary helper. The maximum source horizon
    covers every declared stage. Recurring captures count from the last actual
    publication; explicit targets retain their requested identity after rounding.
    Return the adjusted schedule and the shared learner initializer arguments.
    This reads small setup arrays only and writes no files.
    """
    import numpy as np

    from marl_battlegrounds.training.curriculum import (
        _with_history_capture_rounds,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training.validation import (
        _collection_boundary,  # pyright: ignore[reportPrivateUsage]
    )

    settings = config.qmix or config.pqn or config.ppo
    minimum = int(np.max(prepared.source_configs.max_steps))
    interval = -(-(config.past_capture_interval or 0) // config.num_envs)
    if not config.keep_past or interval:
        targets: tuple[int, ...] = ()
    elif config.history_capture_env_steps is not None:
        targets = tuple(
            value // config.num_envs for value in config.history_capture_env_steps
        )
    else:
        rounds = config.total_env_steps // config.num_envs
        targets = tuple((rounds * index + 19) // 20 for index in range(1, 21))
    if check_spacing and config.keep_past and interval and interval < minimum:
        raise ValueError(
            f"past_capture_interval is {interval} rounds; captures need "
            f"at least {minimum} rounds (one maximum game length)"
        )
    previous = None
    for target in targets if check_spacing else ():
        actual = (
            _collection_boundary(
                target * config.num_envs,
                total_env_steps=config.total_env_steps,
                num_envs=config.num_envs,
                rollout_length=settings.rollout_length,
                initial_rounds=0 if config.pqn is None else config.pqn.initial_rounds,
            )[0]
            // config.num_envs
        )
        if check_spacing and previous is not None and actual - previous < minimum:
            raise ValueError(
                f"History captures would be {actual - previous} rounds apart; "
                f"need at least {minimum}. Use keep_past=0 or a sparser schedule."
            )
        previous = actual
    count = len(targets)
    schedule = _with_history_capture_rounds(schedule, targets)
    capacity = (
        0
        if not config.keep_past
        else config.total_env_steps // config.num_envs // interval
        if interval
        else count
    )
    return schedule, {
        "keep_past": config.keep_past,
        "history_capture_capacity": capacity,
        "minimum_capture_rounds": minimum,
        "capture_interval_rounds": interval,
    }


def _released_config(
    config: TrainConfig, *, commits: Mapping[str, str | None] | None = None
) -> TrainConfig:
    """Turn released names in a config's model fields into exact saved forms.

    Parameters
    ----------
    config : TrainConfig
        Settings as the caller gave them.
    commits : mapping or None, default=None
        Commits a saved run recorded (released_models._recorded_commits). On
        resume they win over the package pins, so a supplied config with typed
        names compares equal to the saved run. None for a new run.

    Returns
    -------
    TrainConfig
        The same object when nothing changes. Otherwise a copy where
        pinned_opponent, initial_actor, slot_diagnostic_actor, every
        validation_opponents entry and every validation_partners value that is
        a typed released name holds its exact ``hf://`` address, and a local
        released-layout folder holds the absolute path of its chosen file.
        Opponent and partner names stay as typed; their bindings are resolved
        by _population_bindings.

    Raises
    ------
    ValueError
        A released name cannot be resolved (see released_models), or a
        pinned_opponent, validation opponent or slot_diagnostic_actor is still
        a relative local path after resolution (saved runs need absolute paths).

    Notes
    -----
    Host-only. May download small index files, never a model file.
    """
    from marl_battlegrounds._method_loading import validate_saved_method_reference
    from marl_battlegrounds.released_models import (
        _saved_reference,  # pyright: ignore[reportPrivateUsage]
    )

    def saved(value: str) -> str:
        """Return one model field's saved form."""
        return _saved_reference(value, commits=commits)

    changes: dict[str, Any] = {}
    for name in ("pinned_opponent", "initial_actor", "slot_diagnostic_actor"):
        value = getattr(config, name)
        if isinstance(value, str) and (resolved := saved(value)) != value:
            changes[name] = resolved
    if config.validation_opponents is not None:
        opponents = tuple(saved(value) for value in config.validation_opponents)
        if opponents != config.validation_opponents:
            changes["validation_opponents"] = opponents
    if config.validation_partners is not None:
        partners = {
            name: saved(value) for name, value in config.validation_partners.items()
        }
        if partners != dict(config.validation_partners):
            changes["validation_partners"] = partners
    result = replace(config, **changes) if changes else config
    for value in (
        result.pinned_opponent,
        result.slot_diagnostic_actor,
        *(result.validation_opponents or ()),
    ):
        if value is not None:
            validate_saved_method_reference(value)
    return result


def _population_bindings(
    selection: Mapping[str, float] | Sequence[str] | None,
    bindings: Mapping[str, Any] | None,
    *,
    saved_names: Sequence[str] | None = None,
    commits: Mapping[str, str | None] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """Resolve declared external members in stable order before output setup.

    Selected references use the ordinary method loader. A live alias wins over
    reference resolution. Extra live bindings remain declared, so a later rule
    can select them without changing the population's structure. Reserved self
    and past names always belong to the learner/history owner. A released
    model name becomes its exact ``hf://`` address in the returned references
    (the saved binding receipt), using commits first when given (a resumed
    run's recorded commits); the member name stays as typed.
    """
    from marl_battlegrounds._method_loading import load_method
    from marl_battlegrounds.released_models import (
        _saved_reference,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training.opponents import opponent_selection_names

    names = opponent_selection_names(selection)
    if bindings is not None and not isinstance(cast(object, bindings), Mapping):
        raise TypeError("opponents must be a mapping of names to live methods")
    supplied = {} if bindings is None else dict(bindings)
    if any(
        not isinstance(cast(object, name), str)
        or not name.strip()
        or name in {"self", "past"}
        for name in supplied
    ):
        raise ValueError("Opponent aliases must be nonempty and cannot be self or past")
    if selection is None:
        if supplied:
            raise ValueError(
                "Live opponents need an opponents selection in TrainConfig"
            )
        return None, {}
    ordered = dict.fromkeys((*names, *supplied) if saved_names is None else saved_names)
    if saved_names is not None and (set(names) | set(supplied)) - {
        "self",
        "past",
        *ordered,
    }:
        raise ValueError("Resume opponent names differ from the saved population")
    original: dict[str, Any] = {}
    for name in ordered:
        if name not in {"self", "past"}:
            value = supplied.get(name, name)
            original[name] = (
                _saved_reference(value, commits=commits)
                if isinstance(value, str)
                else value
            )
    references = {
        name: value for name, value in original.items() if isinstance(value, str)
    }
    return {
        name: load_method(value) if isinstance(value, str) else value
        for name, value in original.items()
    }, references


def _bind_training_population(
    kind: str,
    declared_selection: Mapping[str, float] | Sequence[str] | None,
    bindings: Mapping[str, Any] | None,
    saved: dict[str, Any] | None,
    changes: Mapping[str, Any] | None,
) -> tuple[
    dict[str, Any] | None,
    dict[str, str],
    dict[str, Any],
    dict[str, str],
    Mapping[str, float] | Sequence[str] | None,
]:
    """Resolve opponents or partners while keeping saved members in their slots.

    Return the parent population/references, new child members/references, and
    current selection. Restore uses saved alias order and reference bindings;
    live-only aliases require explicit rebinding. Only a declared child can add
    members. Load all references before output, leaving resource scopes to train.
    Released names given on resume or continuation use the commits the saved
    run recorded, so newer package pins never change a resumed run.
    """
    from marl_battlegrounds.released_models import (
        _recorded_commits,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training.opponents import opponent_selection_names

    commits = None if saved is None else _recorded_commits(saved)

    supplied = {} if bindings is None else dict(bindings)
    saved_members: list[dict[str, Any]] = (
        [] if saved is None else saved["collection"].get(f"{kind}_members", [])
    )
    saved_names = (
        None if saved is None else tuple(member["name"] for member in saved_members)
    )
    selection = (
        declared_selection
        if saved is None
        else saved["metadata"].get(f"{kind}_selection", declared_selection)
    )
    parent_bindings = supplied
    if saved is not None:
        parent_bindings: dict[str, Any] = {}
        for member in saved_members:
            name = member["name"]
            if name in supplied:
                parent_bindings[name] = supplied[name]
            elif member.get("reference") is not None:
                parent_bindings[name] = member["reference"]
            else:
                raise ValueError(f"Resume needs a live {kind} binding for {name!r}")
        if changes is None and set(supplied) - set(parent_bindings):
            raise ValueError(f"Resume cannot add {kind}s; use extend_training")
    population, references = _population_bindings(
        selection, parent_bindings, saved_names=saved_names, commits=commits
    )
    additions: dict[str, Any] = {}
    added_references: dict[str, str] = {}
    if changes is not None:
        future_selection = changes.get(kind + "s", selection)
        future_names = opponent_selection_names(future_selection)
        known = set(() if saved_names is None else saved_names)
        additional_names = dict.fromkeys((*future_names, *supplied))
        additional_bindings = {
            name: supplied.get(name, name)
            for name in additional_names
            if name not in {"self", "past", *known}
        }
        if additional_bindings and future_selection is None:
            raise ValueError(f"Added {kind}s need an explicit selection")
        if additional_bindings:
            resolved, added_references = _population_bindings(
                {name: 1.0 for name in additional_bindings},
                additional_bindings,
                commits=commits,
            )
            additions = resolved or {}

    return population, references, additions, added_references, selection


def _validation_setup(
    config: TrainConfig,
    collection: TrainingCollection,
    bindings: Mapping[str, Any] | None,
    saved: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, str]]:
    """Freeze the deployed validation team, its maps and final-stage rosters.

    Parameters
    ----------
    config : TrainConfig
        The run's settings. config.validation_maps chooses the validation maps;
        None keeps all five held-out maps (42 through 46).
    collection : TrainingCollection
        The prepared training collection. Its partners, learner slots and last
        curriculum stage decide the deployed team and rosters.
    bindings : mapping or None
        Live validation partners from train(validation_partners=...). None
        adds none.
    saved : dict or None, default=None
        The resumed checkpoint's description. When given, its saved partner
        references are rebound and its saved deployment must equal the new one.

    Returns
    -------
    tuple[dict, dict or None, dict]
        ``(options, declaration, references)``. options holds the keyword
        arguments every validation call receives (partners, learner_slots,
        partner_labels, maps and rosters when they differ from the default).
        declaration is the JSON deployment saved with the run, or None when
        every condition is the default; a chosen validation_maps adds
        ``"maps"`` as a sorted list. references maps each extra partner name to
        its reload reference.

    Raises
    ------
    ValueError
        Partner names overlap, labels name an unknown partner, partners lack
        learner_slots, the saved deployment differs (a resume that would play
        other partners, rosters or maps), or slot_diagnostic is used with a
        custom deployment.

    Notes
    -----
    Every declared training partner contributes one familiar row. Additional
    partners keep their declared label, default unknown. Names must be distinct;
    no member replaces another because their display names happen to match.
    References reload through the ordinary loader; live-only extra members need
    rebinding on resume. The saved declaration is checked before recovery writes.
    Default maps leave the declaration unchanged, so runs without
    validation_maps keep their historical task identities. Native scoring
    rules are unchanged. Host-only; extra partner references may be loaded.
    A released model name becomes its exact ``hf://`` address in references,
    using the saved run's recorded commits first on resume.
    """
    from marl_battlegrounds._method_loading import load_method
    from marl_battlegrounds.released_models import (
        _recorded_commits,  # pyright: ignore[reportPrivateUsage]
        _saved_reference,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.tasks import canonical_tournament_rosters
    from marl_battlegrounds.training.validation import freeze_validation_partners

    extras: dict[str, Any] = dict(config.validation_partners or {})
    if saved is not None:
        for name, reference in (
            saved["metadata"].get("validation_partner_references", {}).items()
        ):
            extras.setdefault(name, reference)
    if bindings is not None:
        extras.update(bindings)
    overlap = set(extras) & set(collection.partner_names)
    if overlap:
        raise ValueError(
            "Validation partner names already belong to training partners: "
            f"{sorted(overlap)}"
        )
    commits = None if saved is None else _recorded_commits(saved)
    references = {
        name: _saved_reference(value, commits=commits)
        for name, value in extras.items()
        if isinstance(value, str)
    }
    extras = {
        name: load_method(references[name]) if isinstance(value, str) else value
        for name, value in extras.items()
    }
    labels = dict(config.validation_partner_labels or {})
    if set(labels) - set(extras):
        raise ValueError("Validation partner labels must name an additional partner")
    partners = dict(
        zip(collection.partner_names, collection.partner_systems, strict=True)
    )
    partners.update(extras)
    options: dict[str, Any] = {}
    deployment: dict[str, Any] = {}
    if partners:
        if collection.learner_slots is None:
            raise ValueError(
                "Validation partners require learner_slots and training partners"
            )
        labels = {**dict.fromkeys(collection.partner_names, "familiar"), **labels}
        frozen, deployment = freeze_validation_partners(
            partners, learner_slots=collection.learner_slots, partner_labels=labels
        )
        options.update(
            partners=frozen,
            learner_slots=collection.learner_slots,
            partner_labels=labels,
        )
    schedule = collection.schedule.arrays
    last = int(schedule.stage_count) - 1
    canonical, _ = canonical_tournament_rosters()
    class_ids = schedule.roster_class_ids
    if class_ids is not None and bool(class_ids[last].any()):
        ids = cast(list[int], class_ids[last].tolist())
        rosters = (
            tuple(canonical[value - 1] for value in ids[:5] if value),
            tuple(canonical[value - 1] for value in ids[5:] if value),
        )
    else:
        roster = canonical[: int(schedule.team_sizes[last])]
        rosters = roster, roster
    for key, roster in zip(("system_roster", "opponent_roster"), rosters, strict=True):
        if roster != canonical:
            options[key] = roster
            deployment[key] = list(roster)
    if config.validation_maps is not None:
        # The shared deployment owner carries custom maps into every task,
        # saved record and selection condition.
        options["maps"] = config.validation_maps
        deployment["maps"] = list(config.validation_maps)
    declaration = deployment or None
    if (
        saved is not None
        and saved["metadata"].get("validation_deployment") != declaration
    ):
        raise ValueError(
            "Validation deployment differs from the saved run; "
            "rebind its original partners"
        )
    if config.slot_diagnostic and declaration is not None:
        raise ValueError(
            "slot_diagnostic requires its canonical full-learner roster "
            "and all five validation maps"
        )
    return options, declaration, references


def _local_actor(reference: str) -> str:
    """Return a slot-diagnostic actor's local path, downloading a released one.

    An ``hf://`` address becomes its checked file in Hugging Face's standard
    cache (downloaded once when missing); any other reference, an absolute
    actor folder or model file path, is returned unchanged. Raises ValueError
    when a released file cannot be found or checked.
    """
    if not reference.startswith("hf://"):
        return reference
    from marl_battlegrounds.released_models import (
        _fetch,  # pyright: ignore[reportPrivateUsage]
    )

    return str(_fetch(reference).path)


def _check_panel_purpose(config: TrainConfig, panel: FrozenPanel | None) -> None:
    """Check existing slot and demonstration rules for a chosen frozen panel.

    config is the current run configuration; panel may be absent for ordinary
    development runs. Missing required panels, unqualified demonstrations or
    System-panel slot comparisons without an explicit actor raise ValueError.
    This reads no files and changes no state. The caller checks any actor path.
    """
    if config.slot_diagnostic and panel is None:
        raise ValueError("slot_diagnostic requires a frozen validation panel")
    if (
        config.slot_diagnostic
        and panel is not None
        and panel.schema_version == 2
        and not config.slot_diagnostic_actor
    ):
        raise ValueError(
            "A System panel needs an explicit slot_diagnostic_actor "
            "for this optional comparison"
        )
    if config.purpose == "demonstration" and (panel is None or not panel.qualified):
        raise ValueError("Demonstration requires a qualified frozen validation panel")


def _require_new_output(root: Path, *, locked: bool = False) -> None:
    """Refuse a new or child run folder that holds anything but its lock file.

    Parameters
    ----------
    root : Path
        Resolved output folder of a new run or a child run.
    locked : bool, default=False
        True when the caller already holds root's run lock. The rule is the
        same; only the error message changes, to say that the files appeared
        while this call was setting up.

    Raises
    ------
    ValueError
        root is not a folder, or it holds any entry other than a plain
        ``.training.lock`` file (a folder or link with that name is refused).

    Notes
    -----
    The rule itself is _run_io.unused_run_folder: a missing folder, or one
    holding only the lock file left by a call that stopped before it saved
    anything, passes. _train calls this once before setup, so a used folder
    fails fast, and again while it holds the run lock, before its first
    write, so a call that waited in setup never writes into a run that
    another call saved meanwhile. This reads folder entries only and writes
    nothing.
    """
    if not unused_run_folder(root):
        raise ValueError(
            "output_dir must be new or empty, but files appeared there while "
            "this call was setting up, for example from another training call"
            if locked
            else "output_dir must be new or empty"
        )


@contextmanager
def _checkpoint_memory_scope() -> Generator[None]:
    """Collect unused saved-array cycles before setup and after a training call.

    The inner runner has returned before normal exit cleanup. Failure cleanup
    preserves the exception and its traceback; arrays still owned by that
    traceback or by the caller remain live. This performs host cleanup only.
    """
    gc.collect()
    try:
        yield
    finally:
        gc.collect()


def train(
    config: TrainConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    validation_opponents: Sequence[Any] | None = None,
    opponents: Mapping[str, Any] | None = None,
    partners: Mapping[str, Any] | None = None,
    validation_partners: Mapping[str, Any] | None = None,
    matchmaking: Callable[[dict[str, Any], int], Mapping[str, object]] | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Run or resume one complete declared training experiment on this device.

    Parameters
    ----------
    config : TrainConfig or None, default=None
        Required for a new run. Resume inherits saved settings when omitted;
        an explicit config must match all saved settings exactly.
    output_dir : str or Path or None, default=None
        Exact new or empty run directory. Required for new runs and mutually
        exclusive with resume_from. No generated parent run name is inserted.
        A folder holding only the lock file ``.training.lock``, left by a call
        that stopped before saving anything, counts as empty. train checks the
        folder before setup and again after it takes the run lock, before its
        first write. So when two calls get the same new folder, the one that
        takes the lock first owns it. The other raises RuntimeError while the
        owner still runs, or ValueError once the owner has saved files; if the
        owner stopped before saving anything, the other uses the folder as
        empty. Neither call becomes a resume.
        Inside this source checkout the folder must be ignored by Git; package
        source folders are refused so output cannot change source identity.
    resume_from : str or Path or None, default=None
        Saved run folder or exact complete learner checkpoint directory. A run
        folder follows only its checked latest_checkpoint.json pointer. The
        pointer is read again after the run lock is taken: if another call
        saved a newer checkpoint meanwhile, this call raises ValueError before
        changing any file, so a delayed resume never rolls the run back. An
        exact checkpoint folder is resumed as named, even an older one. Discover
        the containing run, verify scientific/source/content facts, execution
        identity and numerical state before writer rewind. Exports saved after
        the resumed checkpoint (checkpoints, actors and past copies from the
        updates a resume abandons) then move whole under the run's attempts/
        folder, each with an abandoned_export_preserved event, so the run's
        folders hold one trajectory. The saved compiler policy,
        backend, device kind, runtime and numerical settings must match.
        Older checkpoints without that identity remain readable but cannot
        resume. A saved config without ppo.spawn_frame means "world" whatever
        the current default is; a supplied config that disagrees is rejected.
        A checkpoint saved before the Red Zone rule (actor input schema 1) is
        refused before any file is touched; resuming it needs the source
        environment that created it. A supplied config whose red_zone_depth
        differs from the saved run's is refused with a message naming both
        depths, also before any file is touched.
        A QMIX or PQN-VDN resume restores into a shape-only template, so
        restoring never holds a second learner. Once training starts, train
        keeps no reference to the first learner state; a QMIX block then holds
        the previous replay and the new one plus working space (plan for up to
        about three replays). Failed recovery retains its explicit retry
        marker.

    validation_opponents : sequence of System, Policy or str, optional
        Live methods or reload references for a new frozen panel. A new run cannot
        also declare config.validation_opponents. With config.validation_panel or
        resume, these bindings must match every saved member in order; they never
        replace membership. A live-only client must be supplied again on resume.

    opponents : mapping[str, System | Policy | str] or None, default=None
        Live named population bindings for config.opponents. Unbound names use
        the ordinary method loader. Additional bindings may start unused and be
        chosen later by matchmaking. Supply live-only methods again on resume.
        Frozen numerical weights are stored in the learner checkpoint. Repeated
        references share one setup load, including factories; role memory stays
        separate. Rebinding an identical saved artifact may change only its path.
        Older records without an artifact receipt keep their original path check.
    partners : mapping[str, System | Policy | str] or None, default=None
        Frozen named teammates for config.partners. They control the slots
        outside learner_slots; current and past opponents mirror that team with
        their own partner draw. Named external opponents control their whole
        team. Supply live-only bindings again on resume. A child may append
        members when changes.partners names its future mixture or order.
    validation_partners : mapping[str, System | Policy | str] or None, default=None
        Additional named evaluation partners, alongside every training partner.
        Training partners are labelled familiar; additional labels come from
        config.validation_partner_labels and default to unknown. Names must not
        replace a training partner. Saved references reload normally; live-only
        methods need the same binding on resume.
    matchmaking : callable or None, default=None
        Optional rule called between updates as rule(stats, env_steps). Return
        opponent weights/order under "opponents", partner weights/order under
        "partners", and optional weights by stable capture ID under "past".
        stats.members holds cumulative and last-update opponent W/D/L, games
        and points; stats.partners names the frozen teammates. Changes affect
        new games only.
        Add members at an explicit continuation boundary; this callback cannot
        change the population's structure. Exceptions stop through normal recovery.
        Launcher factory code/kwargs receipts are saved. Ordinary resume requires
        the same receipt; an explicit continuation may choose a new callback.
        Live Python callback state remains caller-owned and is not restored.
    on_update : callable or None, default=None
        Receive a copy of each saved training_updates.jsonl row between blocks,
        after its normal file append. Includes run_id, env_steps and existing attempt
        and update counters. No per-step call or extra device work is added.
        Supply it again on resume; it is never saved in config JSON. Exceptions
        stop the run with that row intact and the last recovery checkpoint still
        available. Launcher-created callbacks carry code/kwargs receipts; ordinary
        resume requires the same receipts before recovery writes. Live callbacks
        remain caller-owned and their hidden state is not saved.
        External logging may receive repeated work after recovery;
        use run and step IDs to deduplicate. None skips the copy and call.

    Returns
    -------
    TrainResult
        Saved actor/report paths and exact counts after all declared work ends.
        A development run without a panel has no selected actor. Completion
        records execution, not a claim of useful learned behavior.

    Raises
    ------
    ImportError
        Report dependencies are missing or cannot import. Install the training
        extra. This check runs before learner setup or changes to run files.
    ValueError, TypeError
        Settings, content, checkpoint or saved results are invalid/incompatible,
        or a new run's output_dir holds files, before setup or when checked
        again under the run lock, or a resumed run folder's latest checkpoint
        changed while this call was setting up.
    RuntimeError
        Another writer owns the run or numerical execution fails.
    OSError
        Required output cannot be read or published. Failure state and the last
        complete checkpoint remain available when the filesystem permits it.

    Notes
    -----
    Runs synchronously and owns its files/process lock until completion. Device
    selection belongs to the caller's environment before importing JAX. No
    budget extension, replacement seed or silent retry occurs. Validation uses
    independent frozen Systems and keys, with its own saved evaluation records, and
    plays under the run's red_zone_depth. Setup prepares the training content
    once (checking the depth against every map width) before a new
    validation panel is published, and passes it to the learner; a ranked
    panel whose recorded depth differs from the run's is refused. A new panel
    is published only after the run folder is locked and checked again.
    Unused Python cycles holding saved arrays are collected before setup,
    after saved states are replaced, and when this call exits. This host work
    does not change learner calculations. Arrays held by the caller or a kept
    error traceback remain live.
    """
    from marl_battlegrounds._method_loading import (
        _method_loading_scope,  # pyright: ignore[reportPrivateUsage]
    )

    with _checkpoint_memory_scope(), _method_loading_scope():
        return _train(
            config,
            output_dir=output_dir,
            resume_from=resume_from,
            validation_opponents=validation_opponents,
            opponents=opponents,
            partners=partners,
            validation_partners=validation_partners,
            matchmaking=matchmaking,
            on_update=on_update,
        )


def extend_training(
    checkpoint: str | Path,
    *,
    additional_env_steps: int,
    output_dir: str | Path,
    changes: Mapping[str, object] | None = None,
    validation_opponents: Sequence[Any] | None = None,
    opponents: Mapping[str, Any] | None = None,
    partners: Mapping[str, Any] | None = None,
    validation_partners: Mapping[str, Any] | None = None,
    matchmaking: Callable[[dict[str, Any], int], Mapping[str, object]] | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Continue a full learner checkpoint into a separate child run.

    Parameters
    ----------
    checkpoint : str or Path
        Complete post-Red-Zone learner folder, ``checkpoints/<id>``. Actor
        exports and pruned learner folders cannot continue training. All saved
        learner state, game memory and random streams are restored first.
    additional_env_steps : int
        Positive number of real environment transitions to add to the saved
        step, divisible by the unchanged environment batch. This does not
        restart the parent's curriculum or stretch its learning-rate decay.
    output_dir : str or Path
        Exact new or empty child folder outside the parent run. The parent is
        read only. The child can later use ordinary exact-config resume. As in
        train, a lone ``.training.lock`` counts as empty, and the folder is
        checked again under the run lock before the child's first write, so
        two calls given the same child folder never both write there.
    changes : mapping or None, default=None
        Optional future validation, learning-rate, exploration, seed, curriculum,
        opponent/partner selection, history, actor retention and reward declarations.
        retain_all_actors changes only future child boundaries; old retained actors
        and their provenance stay unchanged. A new seed branches
        library-owned streams once; method-owned opaque randomness is unchanged.
        A new reward definition or stored potential-shaping change clears and
        refills Q experience before learning; optimizer state and statistics
        carry over. Partner/opponent distribution changes retain valid rows.
        All methods allow gamma changes. PPO's ppo mapping may change
        gae_lambda, clip_epsilon,
        entropy_coefficient, value_coefficient and max_grad_norm. PQN's pqn
        mapping may change td_lambda and max_grad_norm. Architecture, optimizer
        layout and batch settings stay fixed; rates use learning_rate only.
        Other keys fail. Existing
        settings remain unchanged by default. PQN needs an explicit rate choice
        if any added optimizer update would use its terminal rate. See the
        continuation example for each supported declaration. A changed validation
        panel may have runtime-only ``bindings``; those clients are checked by
        their saved identities and are never serialized into the declaration.
    validation_opponents : sequence of System, Policy or str, optional
        Runtime bindings for the parent's frozen validation panel, as in train.
        Live-only members must be supplied again. Membership and saved identities
        must match. To change the future panel, use changes.validation.panel and
        its own optional bindings. Neither route changes the parent's panel.

    opponents : mapping[str, System | Policy | str] or None, default=None
        Live named population bindings for config.opponents. Unbound names use
        the ordinary method loader. Additional bindings may start unused and be
        chosen later by matchmaking. Supply live-only methods again on resume.
        Frozen numerical weights are stored in the learner checkpoint. Repeated
        references share one setup load, including factories; role memory stays
        separate. Rebinding an identical saved artifact may change only its path.
        Older records without an artifact receipt keep their original path check.
    partners : mapping[str, System | Policy | str] or None, default=None
        Frozen named teammates for config.partners. They control the slots
        outside learner_slots; current and past opponents mirror that team with
        their own partner draw. Named external opponents control their whole
        team. Supply live-only bindings again on resume. A child may append
        members when changes.partners names its future mixture or order.
    validation_partners : mapping[str, System | Policy | str] or None, default=None
        Additional named evaluation partners, alongside every training partner.
        Training partners are labelled familiar; additional labels come from
        config.validation_partner_labels and default to unknown. Names must not
        replace a training partner. Saved references reload normally; live-only
        methods need the same binding on resume.
    matchmaking : callable or None, default=None
        Optional rule called between updates as rule(stats, env_steps). Return
        opponent weights/order under "opponents", partner weights/order under
        "partners", and optional weights by stable capture ID under "past".
        stats.members holds cumulative and last-update opponent W/D/L, games
        and points; stats.partners names the frozen teammates. Changes affect
        new games only.
        Add members at an explicit continuation boundary; this callback cannot
        change the population's structure. Exceptions stop through normal recovery.
        Launcher factory code/kwargs receipts are saved. Ordinary resume requires
        the same receipt; an explicit continuation may choose a new callback.
        Live Python callback state remains caller-owned and is not restored.
    on_update : callable or None, default=None
        Receive a copy of each saved training_updates.jsonl row between blocks,
        after its normal file append. Includes run_id, env_steps and existing attempt
        and update counters. No per-step call or extra device work is added.
        Supply it again on resume; it is never saved in config JSON. Exceptions
        stop the run with that row intact and the last recovery checkpoint still
        available. Launcher-created callbacks carry code/kwargs receipts; ordinary
        resume requires the same receipts before recovery writes. Live callbacks
        remain caller-owned and their hidden state is not saved.
        External logging may receive repeated work after recovery;
        use run and step IDs to deduplicate. None skips the copy and call.

    Returns
    -------
    TrainResult
        Child paths, cumulative counts and the best eligible saved actor. A
        run without a validation panel or eligible trained candidate has no
        selected actor. A child ending during warmup still exports its actor;
        status.json explains why no checkpoint was selected.

    Raises
    ------
    ImportError
        Report dependencies are missing or cannot import. Install the training
        extra. This check runs before learner setup or changes to run files.
    ValueError, TypeError
        The budget, declaration, full checkpoint or source transition is not
        supported, or output_dir holds files, before setup or when checked
        again under the run lock. Actor loading remains independent of this
        training rule.
    RuntimeError, OSError
        Another writer holds the child folder's lock, or a numerical operation
        or file operation fails. Complete saved child boundaries remain
        available for explicit resume.

    Notes
    -----
    Runs synchronously on the caller's selected device. Saves the parent
    identity and future declaration before taking another training step.
    Parent and child are one training lineage, not independent training seeds.
    Uses the same saved-array cleanup as train. Caller-held arrays and kept
    error tracebacks remain live.
    """
    if type(additional_env_steps) is not int or additional_env_steps <= 0:
        raise ValueError("additional_env_steps must be a positive integer")
    if changes is not None and not isinstance(cast(object, changes), Mapping):
        raise TypeError("changes must be a mapping or None")
    declared = {} if changes is None else dict(changes)
    if unknown := set(declared) - CONTINUATION_CHANGE_KEYS:
        raise ValueError(f"Unsupported continuation changes: {sorted(unknown)}")
    future_bindings = None
    validation_changes = declared.get("validation")
    if isinstance(validation_changes, Mapping):
        validation_changes = dict(cast(Mapping[str, object], validation_changes))
        future_bindings = validation_changes.pop("bindings", None)
        if future_bindings is not None:
            if "panel" not in validation_changes:
                raise ValueError(
                    "Use validation_opponents for the parent's panel; "
                    "changes.validation.bindings requires a declared future panel"
                )
            if isinstance(future_bindings, (str, bytes)) or not isinstance(
                future_bindings, Sequence
            ):
                raise TypeError("Validation bindings must be a sequence of methods")
        declared["validation"] = validation_changes
    # Runtime clients stay outside JSON. Scientific settings remain finite data.
    declared = json.loads(json.dumps(declared, allow_nan=False))
    from marl_battlegrounds._method_loading import (
        _method_loading_scope,  # pyright: ignore[reportPrivateUsage]
    )

    with _checkpoint_memory_scope(), _method_loading_scope():
        return _train(
            output_dir=output_dir,
            resume_from=checkpoint,
            validation_opponents=validation_opponents,
            opponents=opponents,
            partners=partners,
            validation_partners=validation_partners,
            matchmaking=matchmaking,
            on_update=on_update,
            extension={
                "additional_env_steps": additional_env_steps,
                "changes": declared,
                "validation_bindings": future_bindings,
            },
        )


def _child_config(
    value: dict[str, Any], *, total_env_steps: int, root_total_env_steps: int
) -> TrainConfig:
    """Validate inherited settings while allowing a child to end during warmup.

    value contains ordinary saved settings. root_total_env_steps is the verified
    original fresh-run budget; total_env_steps is the child's declared absolute
    end. Fresh-run minimum warmup/curriculum lengths do not apply to a child
    segment. All method, batch, scientific and integer-limit checks still apply.
    This constructs a new frozen config and never mutates a caller's object.
    """
    if type(total_env_steps) is not int or total_env_steps <= 0:
        raise ValueError("Child total_env_steps must be a positive integer")
    if type(root_total_env_steps) is not int or root_total_env_steps <= 0:
        raise ValueError("Continuation requires its original valid training budget")
    checked = config_from_dict(
        {**value, "total_env_steps": max(total_env_steps, root_total_env_steps)}
    )
    if total_env_steps % checked.num_envs or any(
        step > total_env_steps for step in checked.checkpoint_env_steps
    ):
        raise ValueError("Child budget and save points must contain valid whole rounds")
    # Only the fresh-run minimum is different for a checked child. Construct its
    # declared total after the remaining fields passed the ordinary validator.
    object.__setattr__(checked, "total_env_steps", total_env_steps)
    return checked


def _saved_config(details: dict[str, Any]) -> TrainConfig:
    """Read ordinary settings or verify a child's parent before its config.

    Learner descriptions are content identified. A child keeps the parent's
    original budget and actual saved boundary; arbitrary changed settings do
    not gain the child-only warmup allowance. Parent descriptions must remain
    at their recorded paths, even when old payloads are pruned. An unreadable
    parent raises ValueError naming that path. No payload or run file is changed.
    """
    from marl_battlegrounds.training import _checkpoint_metadata as checkpoints

    value = checkpoints.saved_training_config(details)
    context = details["metadata"].get("continuation")
    if context is None:
        return config_from_dict(value)
    if not isinstance(context, dict) or context.get("schema_version") != 1:
        raise ValueError("Unsupported continuation declaration")
    context = cast(dict[str, Any], context)
    parent_path = Path(context["parent_checkpoint"])
    try:
        parent = checkpoints.read_checkpoint_description(parent_path)
    except ValueError as error:
        raise ValueError(
            f"Cannot read the continuation parent at {parent_path}: {error}. "
            "Keep its saved checkpoint folder and description at that path."
        ) from error
    previous = parent["metadata"].get("continuation")
    root = context["segment"]["root_schedule"]
    expected_root = (
        parent["metadata"]["config"]["total_env_steps"]
        if previous is None
        else previous["segment"]["root_schedule"]["total_env_steps"]
    )
    if (
        parent["kind"] != "learner"
        or parent["checkpoint_id"] != context["parent_checkpoint_id"]
        or parent["metadata"]["run_id"] != context["parent_run_id"]
        or parent["metadata"]["source"] != context["parent_source"]
        or context["child_source"] != details["metadata"]["source"]
        or parent["counters"]["env_steps"] != context["start_env_steps"]
        or context["start_env_steps"] + context["additional_env_steps"]
        != value["total_env_steps"]
        or context["resulting_total_env_steps"] != value["total_env_steps"]
        or root["total_env_steps"] != expected_root
        or context["root_run_id"]
        != (
            parent["metadata"]["run_id"]
            if previous is None
            else previous["root_run_id"]
        )
    ):
        raise ValueError("Continuation differs from its saved parent boundary")
    parent_config = checkpoints.saved_training_config(parent)
    parent_config.setdefault("initial_actor", None)
    value.setdefault("initial_actor", None)
    for name, default in (
        ("keep_past", 20),
        ("opponents", None),
        ("partners", None),
        ("reward", None),
        ("past_capture_interval", None),
        ("history_capture_env_steps", None),
    ):
        parent_config.setdefault(name, default)
        value.setdefault(name, default)
    history_changes = context["changes"]
    expected_history = {
        "opponents": history_changes.get("opponents", parent_config["opponents"]),
        "partners": history_changes.get("partners", parent_config["partners"]),
        **{
            name: history_changes.get(name, parent_config[name])
            for name in ("shaping", "shaping_mode", "shaping_coefficient")
        },
        "reward": history_changes.get("reward", parent_config["reward"]),
        "retain_all_actors": history_changes.get(
            "retain_all_actors", parent_config["retain_all_actors"]
        ),
        "keep_past": history_changes.get("keep_past", parent_config["keep_past"]),
        "past_capture_interval": history_changes.get(
            "past_capture_interval",
            None
            if "history_capture_env_steps" in history_changes
            else parent_config["past_capture_interval"],
        ),
        "history_capture_env_steps": history_changes.get(
            "history_capture_env_steps",
            None
            if "past_capture_interval" in history_changes
            else parent_config["history_capture_env_steps"],
        ),
    }
    if any(value[name] != setting for name, setting in expected_history.items()):
        raise ValueError("Continuation history settings differ from its declaration")
    inherited_fields = set(parent_config) - {
        "retain_all_actors",
        "partners",
        "shaping",
        "shaping_mode",
        "shaping_coefficient",
        "opponents",
        "reward",
        "keep_past",
        "past_capture_interval",
        "history_capture_env_steps",
        "total_env_steps",
        "checkpoint_env_steps",
        "validation_panel",
        "validation_opponents",
    }
    expected_points = [
        point
        for point in parent_config["checkpoint_env_steps"]
        if context["start_env_steps"] < point <= value["total_env_steps"]
    ]
    if (
        set(value) != set(parent_config)
        or any(value[name] != parent_config[name] for name in inherited_fields)
        or value["checkpoint_env_steps"] != expected_points
        or value["validation_panel"]
        != details["metadata"]["validation_declaration"]["panel_path"]
        or value["validation_opponents"]
        != (
            None if value["validation_panel"] else parent_config["validation_opponents"]
        )
    ):
        raise ValueError("Continuation changed an inherited training setting")
    return _child_config(
        value,
        total_env_steps=value["total_env_steps"],
        root_total_env_steps=expected_root,
    )


def _require_training_reports() -> None:
    """Check report dependencies before training setup or recovery changes.

    Import plotting support without creating a figure. Every complete trainer
    writes reports, so a missing or broken import raises an actionable ImportError
    before learner setup, output creation or recovery mutation. Configuration
    parsing and launcher help never call this execution-only check.
    """
    from importlib import import_module

    try:
        import_module("matplotlib.pyplot")
    except ImportError as error:
        raise ImportError(
            "Training reports need Matplotlib and its dependencies. "
            "Install marl-battlegrounds[training]."
        ) from error


def _train(
    config: TrainConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    validation_opponents: Sequence[Any] | None = None,
    opponents: Mapping[str, Any] | None = None,
    partners: Mapping[str, Any] | None = None,
    validation_partners: Mapping[str, Any] | None = None,
    matchmaking: Callable[[dict[str, Any], int], Mapping[str, object]] | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    extension: dict[str, Any] | None = None,
) -> TrainResult:
    """Share setup, checked restore and execution for training and a declared fork.

    Public callers use train or extend_training. extension is their normalized
    added budget and future-only changes; it never relaxes ordinary resume.
    Runtime records include setup_stage_seconds: host wall times for named setup
    owners. They add no device synchronization and are not isolated kernel costs.

    A new or child run checks output_dir with _require_new_output before setup
    and again after taking the run lock, before its first write (a new
    validation panel included). A call that waited in setup therefore never
    writes into, or resumes, a run that another call saved meanwhile. Ordinary
    resume skips both checks. It locks the existing run and resolves
    resume_from again before its first write; a run folder whose latest
    checkpoint changed in between raises ValueError instead of rolling back.
    """
    if on_update is not None and not callable(on_update):
        raise TypeError("on_update must be callable or None")
    if matchmaking is not None and not callable(matchmaking):
        raise TypeError("matchmaking must be callable or None")
    from marl_battlegrounds.training.callbacks import (
        _callback_identities,  # pyright: ignore[reportPrivateUsage]
    )

    callback_inputs = _callback_identities(on_update=on_update, matchmaking=matchmaking)
    if output_dir is not None:
        from marl_battlegrounds.evaluation.revision import (
            _check_output_path,  # pyright: ignore[reportPrivateUsage]
        )

        _check_output_path(Path(output_dir))
    _require_training_reports()
    attempt_started = time.monotonic()
    # Host wall times include any device waits inside each setup owner.
    # They are not isolated kernel timings and add no new synchronization.
    setup_seconds: dict[str, float] = {}
    started_at = utc_now()
    from marl_battlegrounds.training import checkpoints, learner, validation
    from marl_battlegrounds.training._compilation import execution_identity
    from marl_battlegrounds.training._content import prepare_training_content
    from marl_battlegrounds.training.curriculum import make_training_schedule

    if extension is None and (output_dir is None) == (resume_from is None):
        raise ValueError("Supply exactly one of output_dir or resume_from")
    if extension is not None and (output_dir is None or resume_from is None):
        raise ValueError("Continuation requires a parent checkpoint and child output")
    saved: dict[str, Any] | None = None
    if extension is not None:
        checkpoint = Path(str(resume_from)).resolve()
    elif resume_from is not None:
        from marl_battlegrounds.training._checkpoint_metadata import (
            read_resume_checkpoint,
        )

        checkpoint, _ = read_resume_checkpoint(resume_from)
    else:
        checkpoint = None
    if checkpoint is not None:
        try:
            saved = checkpoints.read_checkpoint_description(checkpoint)
        except (ValueError, FileNotFoundError) as error:
            if extension is None:
                raise
            raise ValueError(
                "Continuation requires a complete learner checkpoint at "
                "checkpoints/<id>, including actor and learner payloads; "
                "actor exports and pruned folders cannot continue training"
            ) from error
        if saved["kind"] != "learner":
            raise ValueError(
                "Continuation requires a complete learner checkpoint at "
                "checkpoints/<id>; "
                "actors/<id> contains inference weights only"
                if extension is not None
                else "Resume requires a complete learner checkpoint"
            )
        # A checkpoint saved before Red Zone cannot resume here; refuse it
        # before any file (lock, panel or log) is touched.
        checkpoints._require_current_schemas(saved)  # pyright: ignore[reportPrivateUsage]
        if (
            extension is None
            and "callback_inputs" in saved["metadata"]
            and saved["metadata"]["callback_inputs"] != callback_inputs
        ):
            raise ValueError(
                "Resume callback factories or kwargs differ from the saved run; "
                "supply the same runtime callbacks or use an explicit continuation"
            )
        inherited = _saved_config(saved)
        if config is not None and config.red_zone_depth != inherited.red_zone_depth:
            raise ValueError(
                f"Resume config declares red_zone_depth {config.red_zone_depth}, "
                f"but the saved run uses {inherited.red_zone_depth}. The depth is "
                "fixed for a run; start a new run to use another depth."
            )
        if config is not None:
            from marl_battlegrounds.released_models import (
                _recorded_commits,  # pyright: ignore[reportPrivateUsage]
            )

            config = _released_config(config, commits=_recorded_commits(saved))
        if config is not None and config_to_dict(config) != config_to_dict(inherited):
            raise ValueError("Resume config differs from the saved experiment")
        config = inherited
        root = checkpoint.parent.parent
        if (
            checkpoint.parent.name != "checkpoints"
            or not (root / "run_details.json").is_file()
        ):
            raise ValueError("Checkpoint is outside its original training run")
    else:
        if not isinstance(config, TrainConfig):
            raise TypeError("A new run requires TrainConfig")
        # Saved settings hold exact addresses, never names whose files can move.
        config = _released_config(config)
        root = Path(str(output_dir)).resolve()
        _require_new_output(root)
    assert config is not None
    parent_root = root
    if extension is not None:
        root = Path(str(output_dir)).resolve()
        if root == parent_root or parent_root in root.parents:
            raise ValueError("Child output_dir must be outside the parent training run")
        _require_new_output(root)
        if extension["additional_env_steps"] % config.num_envs:
            raise ValueError("additional_env_steps must divide by num_envs exactly")
    if (
        saved is None
        and validation_opponents is not None
        and config.validation_opponents is not None
    ):
        raise ValueError("Declare validation_opponents once, in config or train")
    declared = (
        validation_opponents
        if validation_opponents is not None
        else config.validation_opponents
    )
    population, opponent_references, additions, added_references, selection = (
        _bind_training_population(
            "opponent",
            config.opponents,
            opponents,
            saved,
            None if extension is None else extension["changes"],
        )
    )
    (
        partner_population,
        partner_references,
        partner_additions,
        added_partner_references,
        partner_selection,
    ) = _bind_training_population(
        "partner",
        config.partners,
        partners,
        saved,
        None if extension is None else extension["changes"],
    )
    saved_continuation = (
        None if saved is None else saved["metadata"].get("continuation")
    )
    setup_config = config
    if saved_continuation is not None:
        # A child may end before fresh-run warmup or curriculum minimums. Use
        # its original valid budget only to allocate the restore template;
        # the exact saved child schedule and state replace it before any action.
        setup_config = replace(
            config,
            total_env_steps=max(
                config.total_env_steps,
                saved_continuation["segment"]["root_schedule"]["total_env_steps"],
            ),
        )
    schedule = make_training_schedule(
        total_env_steps=setup_config.total_env_steps,
        num_envs=setup_config.num_envs,
        curriculum=setup_config.curriculum,
        score_threshold_curriculum=setup_config.score_threshold_curriculum,
        early_history_capture=setup_config.pinned_opponent_share > 0,
    )
    # Prepare the content (every map checks the depth against its width)
    # before a new panel can be published, so a bad setting leaves no file.
    setup_stage_started = time.monotonic()
    prepared = prepare_training_content(
        score_thresholds=schedule.score_thresholds,
        red_zone_depth=config.red_zone_depth,
    )
    setup_seconds["content"] = time.monotonic() - setup_stage_started
    schedule, history_options = _history_setup(
        setup_config, schedule, prepared, check_spacing=saved is None
    )
    if saved is not None and "opponent_rows" in saved["collection"]:
        history_options["history_capture_capacity"] = saved["collection"].get(
            "history_capture_capacity", saved["collection"]["opponent_rows"] - 2
        )
    from marl_battlegrounds.training.shaping import resolve_reward

    if extension is not None and "reward" in extension["changes"]:
        assert saved is not None
        reward_identity = saved["collection"].get("reward_identity")
        from functools import partial

        from marl_battlegrounds.training._checkpoint_metadata import (
            _reward_component_keys,  # pyright: ignore[reportPrivateUsage]
        )

        reward = (
            None
            if reward_identity is None
            else partial(
                _restored_reward_shape,
                component_keys=_reward_component_keys(reward_identity),
            )
        )
    else:
        reward, reward_identity = resolve_reward(config.reward)
    initial_actor = None
    initial_origin = None
    if saved is None and config.initial_actor is not None:
        from marl_battlegrounds.training._content import pinned_opponent_evidence

        settings = config.qmix or config.pqn or config.ppo
        initial_system, initial_details = checkpoints.load_initial_actor(
            config.initial_actor,
            method=config.method,
            input_scale=settings.input_scale,
            spawn_frame=settings.spawn_frame,
            parameter_sharing=settings.parameter_sharing,
            normalization="batch_norm"
            if config.pqn is None
            else config.pqn.normalization,
        )
        variables = initial_system.variables
        initial_actor = (
            variables.params
            if config.qmix is not None
            else variables.network
            if config.pqn is not None
            else variables
        )
        initial_export = Path(config.initial_actor)
        if config.initial_actor.startswith("hf://"):
            from marl_battlegrounds.released_models import (
                _fetch,  # pyright: ignore[reportPrivateUsage]
            )

            initial_export = cast(Path, _fetch(config.initial_actor).path)
        initial_origin = {
            "path": config.initial_actor
            if config.initial_actor.startswith("hf://")
            else str(Path(config.initial_actor).resolve()),
            "actor_digest": initial_details["actor_digest"],
            "checkpoint_id": initial_details["checkpoint_id"],
            "evidence": pinned_opponent_evidence(
                prepared.binding, initial_system, export=initial_export
            ),
        }
    # Freeze identities before allocation; publish only after roster preflight.
    new_panel_content: dict[str, Any] | None = None
    panel_path = (
        Path(config.validation_panel)
        if config.validation_panel
        else parent_root / "validation_panel" / "panel.json"
    )
    if config.validation_panel or (saved is not None and panel_path.exists()):
        panel = validation.load_panel(
            panel_path, bindings=declared, red_zone_depth=config.red_zone_depth
        )
    elif declared is not None:
        if saved is not None:
            raise ValueError("Resume cannot introduce a new validation panel")
        panel, new_panel_content = validation._prepare_system_panel(  # pyright: ignore[reportPrivateUsage]
            declared, output_dir=panel_path.parent, roots=None, ranking=None, size=None
        )
    else:
        panel = None
    _check_panel_purpose(config, panel)
    if config.slot_diagnostic and config.slot_diagnostic_actor:
        slot_actor = _local_actor(config.slot_diagnostic_actor)
        if Path(slot_actor).is_file():
            from marl_battlegrounds.portable_models import read_portable_model

            read_portable_model(slot_actor)
        else:
            checkpoints.artifact_identity(slot_actor)

    with ExitStack() as stack:
        from marl_battlegrounds.evaluation.policy_execution import System

        opened_members: set[int] = set()
        for member in (
            *(() if population is None else population.values()),
            *additions.values(),
            *(() if partner_population is None else partner_population.values()),
            *partner_additions.values(),
        ):
            if (
                isinstance(member, System)
                and member.resource_scope is not None
                and id(member) not in opened_members
            ):
                stack.enter_context(member.resource_scope(config.recording))
                opened_members.add(id(member))
        # Setup performs no real action. Resume replaces all template numerical values.
        setup_stage_started = time.monotonic()
        state: Any
        if config.qmix is not None:
            from marl_battlegrounds.training import qmix_learner

            collection, state = qmix_learner.init_qmix_learner(
                schedule=schedule,
                seed=config.seed,
                initial_actor=initial_actor,
                prepared=prepared,
                shaping=config.shaping,
                shaping_coefficient=config.shaping_coefficient,
                shaping_mode=config.shaping_mode,
                metrics=config.metrics,
                recording=config.recording,
                pinned_opponent_share=config.pinned_opponent_share,
                pinned_opponent=config.pinned_opponent,
                opponent_population=population,
                opponent_selection=selection,
                learner_slots=config.learner_slots,
                partner_population=partner_population,
                partner_selection=partner_selection,
                keep_past=history_options["keep_past"],
                history_capture_capacity=history_options["history_capture_capacity"],
                minimum_capture_rounds=history_options["minimum_capture_rounds"],
                capture_interval_rounds=history_options["capture_interval_rounds"],
                reward=reward,
                reward_identity=reward_identity,
                qmix=config.qmix,
            )
        elif config.pqn is not None:
            from marl_battlegrounds.training import pqn_learner

            collection, state = pqn_learner.init_pqn_learner(
                schedule=schedule,
                seed=config.seed,
                initial_actor=initial_actor,
                prepared=prepared,
                shaping=config.shaping,
                shaping_coefficient=config.shaping_coefficient,
                shaping_mode=config.shaping_mode,
                metrics=config.metrics,
                recording=config.recording,
                pinned_opponent_share=config.pinned_opponent_share,
                pinned_opponent=config.pinned_opponent,
                opponent_population=population,
                opponent_selection=selection,
                learner_slots=config.learner_slots,
                partner_population=partner_population,
                partner_selection=partner_selection,
                keep_past=history_options["keep_past"],
                history_capture_capacity=history_options["history_capture_capacity"],
                minimum_capture_rounds=history_options["minimum_capture_rounds"],
                capture_interval_rounds=history_options["capture_interval_rounds"],
                reward=reward,
                reward_identity=reward_identity,
                pqn=config.pqn,
            )
        else:
            collection, state = learner.init_learner(
                schedule=schedule,
                seed=config.seed,
                initial_actor=initial_actor,
                prepared=prepared,
                ppo=config.ppo,
                method=config.method,
                shaping=config.shaping,
                shaping_coefficient=config.shaping_coefficient,
                shaping_mode=config.shaping_mode,
                metrics=config.metrics,
                recording=config.recording,
                pinned_opponent_share=config.pinned_opponent_share,
                pinned_opponent=config.pinned_opponent,
                opponent_population=population,
                opponent_selection=selection,
                learner_slots=config.learner_slots,
                partner_population=partner_population,
                partner_selection=partner_selection,
                keep_past=history_options["keep_past"],
                history_capture_capacity=history_options["history_capture_capacity"],
                minimum_capture_rounds=history_options["minimum_capture_rounds"],
                capture_interval_rounds=history_options["capture_interval_rounds"],
                reward=reward,
                reward_identity=reward_identity,
            )
        setup_seconds["learner_initialization"] = time.monotonic() - setup_stage_started
        from marl_battlegrounds.training.collection import (
            _bind_opponent_references,  # pyright: ignore[reportPrivateUsage]
            _bind_partner_references,  # pyright: ignore[reportPrivateUsage]
        )

        collection = _bind_opponent_references(collection, opponent_references)
        collection = _bind_partner_references(collection, partner_references)
        if saved is not None and saved["metadata"].get("recording") is not None:
            from marl_battlegrounds.training.collection import (
                _opponent_component_table,  # pyright: ignore[reportPrivateUsage]
            )

            policies = saved["metadata"]["recording"]["policies"]
            collection = replace(
                collection,
                actor=replace(
                    collection.actor, components=tuple(policies["team_a"]["components"])
                ),
                opponent=replace(
                    collection.opponent,
                    components=tuple(policies["team_b"]["components"]),
                ),
            )
            collection = _opponent_component_table(collection, state.carry)
        if saved_continuation is not None:
            import jax

            from marl_battlegrounds.training.collection import (
                _expanded_stage_sources,  # pyright: ignore[reportPrivateUsage]
            )
            from marl_battlegrounds.training.curriculum import (
                _restore_continuation_schedule,  # pyright: ignore[reportPrivateUsage]
            )

            schedule = _restore_continuation_schedule(saved_continuation["segment"])
            binding, tracking = _expanded_stage_sources(
                collection, state.carry, schedule
            )
            carry = state.carry._replace(schedule=schedule.arrays, tracking=tracking)
            state = state._replace(carry=carry)
            collection = replace(
                collection,
                binding=binding,
                schedule=schedule,
                carry_spec=jax.eval_shape(_identity, carry),
                root_bits=tuple(
                    saved_continuation.get("key_root_bits", collection.root_bits)
                ),
            )
            from marl_battlegrounds.training._continuation_schedules import (
                continuation_collection,
                schedule_continuation,
            )

            collection = continuation_collection(
                collection, schedule_continuation(schedule)
            )
        validation_options, deployment, validation_references = _validation_setup(
            config, collection, validation_partners, saved
        )
        if config.random_initialization_result is not None:
            from marl_battlegrounds.baselines.qmix import QMIX_TIE_RULE
            from marl_battlegrounds.evaluation.recording_identity import tree_digest

            current = state.carry.history.current_variables
            settings = config.qmix or config.pqn or config.ppo
            payload = (
                current.params
                if config.qmix is not None
                else checkpoints._pqn_actor_item(current.network)  # pyright: ignore[reportPrivateUsage]
                if config.pqn is not None
                else current
            )
            initial = {
                "kind": "actor",
                "actor_digest": tree_digest(payload),
                "input_scale": settings.input_scale,
                "spawn_frame": settings.spawn_frame,
                "schemas": checkpoints.checkpoint_schemas(config.method),
                "parameter_sharing": settings.parameter_sharing,
            }
            if not is_ppo_method(config.method):
                initial.update(epsilon=0.0, tie_rule=QMIX_TIE_RULE)
            initial_digest = checkpoints._inference_digest(initial)  # pyright: ignore[reportPrivateUsage]
            validation.read_random_initialization(
                config.random_initialization_result,
                actor_digest=initial_digest,
                seed_pairs=cast(int, config.random_diagnostic_seed_pairs),
                red_zone_depth=config.red_zone_depth,
                deployment=deployment,
            )
        # PPO keeps the argument-free call; QMIX also records Flashbax, PQN-VDN
        # records the same set as PPO.
        identity = (
            checkpoints.runtime_identity()
            if is_ppo_method(config.method)
            else checkpoints.runtime_identity(method=config.method)
        )
        source = cast(dict[str, Any], identity["source"])
        dependencies = identity["dependencies"]
        runtime = _runtime_details(str(source["package_version"]), config.num_envs)
        runtime["setup_stage_seconds"] = setup_seconds
        execution_identity_now = execution_identity(runtime=runtime["provenance"])
        extension_state = None
        if extension is not None:
            assert checkpoint is not None and saved is not None
            import jax

            compatibility = checkpoints.continuation_source_compatibility(
                saved["metadata"]["source"], source
            )
            collection = checkpoints._check_collection_compatibility(  # pyright: ignore[reportPrivateUsage]
                saved, collection
            )
            template = jax.eval_shape(_identity, state)
            state = None
            setup_stage_started = time.monotonic()
            parent_restore = checkpoints.restore_checkpoint(
                checkpoint,
                collection,
                template,
                expected_metadata={
                    "config": config_to_dict(config),
                    "source": source,
                    "dependencies": dependencies,
                    "execution": execution_identity_now,
                },
                ppo=config.ppo,
                method=config.method,
                qmix=config.qmix,
                pqn=config.pqn,
                _source_compatibility=compatibility,
            )
            setup_seconds["checkpoint_restore"] = time.monotonic() - setup_stage_started
            setup_stage_started = time.monotonic()
            validate_host_state(
                parent_root,
                parent_restore.details,
                history_capture_count=int(
                    parent_restore.state.carry.history.next_capture_id
                ),
                panel=panel,
                validation_options=validation_options,
            )
            setup_seconds["saved_host_validation"] = (
                time.monotonic() - setup_stage_started
            )
            setup_stage_started = time.monotonic()
            config, collection, state, panel, declarations = _prepare_extension(
                config,
                collection,
                parent_restore,
                panel,
                extension,
                source,
            )
            if additions or "opponents" in extension["changes"]:
                from marl_battlegrounds.training.collection import (
                    append_training_opponents,
                )

                collection, carry = append_training_opponents(
                    collection,
                    state.carry,
                    additions,
                    selection=extension["changes"].get("opponents"),
                )
                collection = _bind_opponent_references(collection, added_references)
                state = state._replace(carry=carry)
            if partner_additions or "partners" in extension["changes"]:
                from marl_battlegrounds.training.collection import (
                    append_training_partners,
                )

                collection, carry = append_training_partners(
                    collection,
                    state.carry,
                    partner_additions,
                    selection=extension["changes"].get("partners"),
                )
                collection = _bind_partner_references(
                    collection, added_partner_references
                )
                state = state._replace(carry=carry)
            previous_validation_references = validation_references
            validation_options, deployment, validation_references = _validation_setup(
                config,
                collection,
                {
                    name: member
                    for name, member in validation_options.get("partners", {}).items()
                    if name not in collection.partner_names
                },
            )
            validation_references = {
                **previous_validation_references,
                **validation_references,
            }
            from marl_battlegrounds.training._selection_evidence import (
                freeze_inherited_candidates,
            )

            if deployment is not None:
                declarations["validation_deployment"] = deployment
            if validation_references:
                declarations["validation_partner_references"] = validation_references
            declarations["validation_declaration"] = (
                validation.run_validation_declaration(
                    config_to_dict(config),
                    panel,
                    continuation=declarations["continuation"],
                    deployment=deployment,
                )
            )
            declarations["continuation"]["inherited_candidates"] = (
                freeze_inherited_candidates(
                    checkpoint, declaration=declarations["validation_declaration"]
                )
            )
            declarations["continuation"]["source_compatibility"] = compatibility
            setup_seconds["continuation_preparation"] = (
                time.monotonic() - setup_stage_started
            )
            extension_state = parent_restore, declarations
        preflight_options = {
            name: value
            for name, value in validation_options.items()
            if name != "partner_labels"
        }
        if panel is not None and deployment is not None and panel.schema_version != 2:
            raise ValueError(
                "Custom validation conditions need a panel made with opponents="
            )
        if (
            deployment is not None
            or config.random_diagnostic_seed_pairs is not None
            or (panel is not None and panel.schema_version == 2)
        ):
            methods = (
                panel.methods or tuple(str(member.path) for member in panel.members)
                if panel is not None and panel.schema_version == 2
                else ("random",)
            )
            validation.prepare_validation_teams(
                collection.learner_actor or collection.actor,
                methods,
                red_zone_depth=config.red_zone_depth,
                **preflight_options,
            )
        prepared_recording = None
        if extension_state is not None and config.recording:
            prepared_recording = _prepare_extension_recording(
                extension_state[0], state, collection
            )
            stack.callback(prepared_recording.close)
        if saved is not None and extension_state is None:
            collection = checkpoints._check_collection_compatibility(  # pyright: ignore[reportPrivateUsage]
                saved, collection
            )
        from marl_battlegrounds._method_loading import (
            _release_setup_inputs,  # pyright: ignore[reportPrivateUsage]
        )

        _release_setup_inputs()
        # These temporary dictionaries contain unfrozen factory results. The
        # checked collection/panel/deployment and ExitStack now own each role.
        population = partner_population = None
        additions = partner_additions = {}
        # A child recording is fully checked before even its folder is created.
        root.mkdir(parents=True, exist_ok=True)
        stack.enter_context(run_lock(root))
        if saved is None or extension is not None:
            # The first check ran before setup. Another call may have saved a
            # run here since then, so check again before this call's first write.
            _require_new_output(root, locked=True)
        else:
            # A run folder named its latest checkpoint before setup. Another
            # call may have trained the run further since then; resuming the
            # older checkpoint now would roll the run back.
            from marl_battlegrounds.training._checkpoint_metadata import (
                read_resume_checkpoint,
            )

            assert resume_from is not None and checkpoint is not None
            if read_resume_checkpoint(resume_from)[0] != checkpoint:
                raise ValueError(
                    "resume_from now names a newer checkpoint: another call "
                    "trained this run while this call was setting up. Resume "
                    "again to continue from the newest checkpoint."
                )
        if new_panel_content is not None:
            # Publish under the lock, so a refused call writes no panel.
            assert panel is not None
            validation._publish_system_panel(panel, new_panel_content)  # pyright: ignore[reportPrivateUsage]
        if extension_state is not None:
            parent_restore, declarations = extension_state
            owner = _start_extension(
                root=root,
                config=config,
                collection=collection,
                state=state,
                panel=panel,
                restored=parent_restore,
                declarations=declarations,
                source=source,
                dependencies=dependencies,
                execution=execution_identity_now,
                runtime=runtime,
                started_at=started_at,
                attempt_started=attempt_started,
                stack=stack,
                prepared_recording=prepared_recording,
                validation_options=validation_options,
                on_update=on_update,
                matchmaking=matchmaking,
            )
            state = parent_restore = extension_state = None
            return owner.execute()
        writer = None
        metadata: dict[str, Any]
        restored: checkpoints.RestoredCheckpoint | None = None
        if saved is not None:
            assert checkpoint is not None
            if not is_ppo_method(config.method):
                import jax

                # Release the fresh learner before restore reads the saved one.
                state = jax.eval_shape(_identity, state)
            setup_stage_started = time.monotonic()
            restored = checkpoints.restore_checkpoint(
                checkpoint,
                collection,
                state,
                expected_metadata={
                    "config": config_to_dict(config),
                    "source": source,
                    "dependencies": dependencies,
                    "execution": execution_identity_now,
                },
                ppo=config.ppo,
                method=config.method,
                qmix=config.qmix,
                pqn=config.pqn,
            )
            setup_seconds["checkpoint_restore"] = time.monotonic() - setup_stage_started
            metadata = dict(restored.details["metadata"])
            cursors = metadata.get("log_cursors", {})
            if set(cursors) != {"training_updates.jsonl"}:
                raise ValueError("Checkpoint has no valid training-log recovery cursor")
            validate_log_cursor(
                root / "training_updates.jsonl", cursors["training_updates.jsonl"]
            )
            run_details = json.loads((root / "run_details.json").read_text())
            if run_details.get("selection_rule", "saved") != metadata.get(
                "selection_rule", "saved"
            ):
                raise ValueError("Saved checkpoint selection rule differs from its run")
            if run_details["run_id"] != metadata["run_id"]:
                raise ValueError("Checkpoint belongs to a different training run")
            if run_details.get("schemas") != restored.details["schemas"]:
                raise ValueError("Run and checkpoint schema versions differ")
            if run_details.get("execution") != metadata["execution"]:
                raise ValueError("Run and checkpoint execution identities differ")
            if run_details["panel_digest"] != (None if panel is None else panel.digest):
                raise ValueError("Frozen validation panel differs from the saved run")
            state = restored.state
            host = metadata["host_state"]
            if host["env_steps"] != int(
                state.carry.progress.rounds
            ) * config.num_envs or host["completed_updates"] != int(
                state.completed_updates
            ):
                raise ValueError("Checkpoint host counters differ from learner state")
            if config.qmix is not None:
                from marl_battlegrounds.training.qmix_learner import (
                    _bank_size,  # pyright: ignore[reportPrivateUsage]
                )

                exposure = host.get("sampled_exposure")
                if not isinstance(exposure, dict) or len(
                    cast(list[int], cast(dict[str, Any], exposure).get("by_source", []))
                ) != _bank_size(state.carry):
                    raise ValueError(
                        "Saved sampled exposure does not match the source bank"
                    )
            if config.pqn is not None:
                from marl_battlegrounds.training.pqn_learner import (
                    _bank_size as _pqn_bank_size,  # pyright: ignore[reportPrivateUsage]
                )

                exposure = host.get("used_exposure")
                if not isinstance(exposure, dict) or len(
                    cast(list[int], cast(dict[str, Any], exposure).get("by_source", []))
                ) != _pqn_bank_size(state.carry):
                    raise ValueError(
                        "Saved used exposure does not match the source bank"
                    )
            if is_ppo_method(config.method) and "ppo_exposure" in host:
                expected_sources = len(collection.binding.source_configurations)
                if any(
                    len(host["ppo_exposure"].get(f"{kind}_source", []))
                    != expected_sources
                    for kind in ("policy", "value")
                ):
                    raise ValueError(
                        "Saved PPO exposure does not match the source bank"
                    )
            validate_host_state(
                root,
                restored.details,
                history_capture_count=int(restored.state.carry.history.next_capture_id),
                panel=panel,
                validation_options=validation_options,
            )
            if (root / "status.json").is_file():
                prior_status = json.loads((root / "status.json").read_text())
                prior_elapsed = prior_status.get("elapsed_seconds", 0.0)
                if (
                    prior_status.get("run_id") == metadata["run_id"]
                    and isinstance(prior_elapsed, (int, float))
                    and math.isfinite(prior_elapsed)
                ):
                    host["elapsed_seconds"] = max(
                        host["elapsed_seconds"], prior_elapsed
                    )
            writer = checkpoints.resume_recording(restored, root)
            if writer is not None:
                stack.callback(writer.close)
            for name in ("checkpoints", "actors", "past_copies"):
                for cleanup in checkpoints._cleanup_pending_writes(root / name):  # pyright: ignore[reportPrivateUsage]
                    append_jsonl(
                        root / "run_events.jsonl",
                        {
                            "event": "pending_write_cleanup",
                            "time_utc": utc_now(),
                            "attempt_id": metadata["attempt_id"],
                            **cleanup,
                        },
                        durable=True,
                    )
            for old, new in set_aside_abandoned_exports(
                root, host["env_steps"], root / "attempts" / metadata["attempt_id"]
            ):
                append_jsonl(
                    root / "run_events.jsonl",
                    {
                        "event": "abandoned_export_preserved",
                        "time_utc": utc_now(),
                        "attempt_id": metadata["attempt_id"],
                        "path": str(old),
                        "preserved_path": str(new),
                        "restore_checkpoint": saved["checkpoint_id"],
                    },
                    durable=True,
                )
            abandoned = preserve_log_suffix(
                root / "training_updates.jsonl",
                cursors["training_updates.jsonl"],
                root / "attempts" / metadata["attempt_id"],
            )
            if abandoned is not None:
                append_jsonl(
                    root / "run_events.jsonl",
                    {
                        "event": "abandoned_updates_preserved",
                        "time_utc": utc_now(),
                        "attempt_id": metadata["attempt_id"],
                        "path": str(abandoned),
                        "restore_checkpoint": saved["checkpoint_id"],
                    },
                    durable=True,
                )
            restore_log_cursor(
                root / "training_updates.jsonl", cursors["training_updates.jsonl"]
            )
            _preserve_reports(root, metadata["attempt_id"])
            atomic_json(
                root / "validation_results.json",
                host.get("routine_results", []) + host.get("confirmation_results", []),
            )
            if config.random_diagnostic_seed_pairs is not None:
                atomic_json(root / "random_diagnostics.json", host["random_results"])
            atomic_json(root / "selection.json", host.get("selection"))
            atomic_json(root / "exposure.json", None)
            atomic_json(root / "slot_diagnostic.json", None)
            # Publish unfinished status before clearing recovery ownership. A
            # crash before the new attempt starts must not revive old completion.
            atomic_json(
                root / "status.json",
                {
                    "schema_version": 1,
                    "run_id": metadata["run_id"],
                    "attempt_id": metadata["attempt_id"],
                    "status": "incomplete",
                    "phase": "resuming",
                    "env_steps": host["env_steps"],
                    "total_env_steps": config.total_env_steps,
                    "completed_updates": host["completed_updates"],
                    "elapsed_seconds": host["elapsed_seconds"],
                    "latest_checkpoint": str(checkpoint),
                    "pending_task": host["pending"],
                    "process": process_identity(),
                    "updated_at": utc_now(),
                },
            )
            checkpoints.finish_checkpoint_recovery(restored, root)
            metadata["parent_checkpoint"] = saved["checkpoint_id"]
            metadata["attempt_id"] = uuid.uuid4().hex
        else:
            metadata = {
                "run_id": uuid.uuid4().hex,
                "attempt_id": uuid.uuid4().hex,
                "parent_checkpoint": None,
                "selection_rule": "point_margin",
                "config": config_to_dict(config),
                "source": source,
                "dependencies": dependencies,
                "execution": execution_identity_now,
                "host_state": {},
                "initial_actor": initial_origin,
                "callback_inputs": callback_inputs,
                "recording": None,
            }
            if deployment is not None:
                metadata["validation_deployment"] = deployment
                metadata["validation_declaration"] = (
                    validation.run_validation_declaration(
                        config_to_dict(config), panel, deployment=deployment
                    )
                )
            if validation_references:
                metadata["validation_partner_references"] = validation_references
            if config.recording:
                from marl_battlegrounds.evaluation.run_writer import RunWriter

                policies = _training_registrations(collection)
                # The pass names the pinned System; the writer and the
                # checkpoint must hold the same details, which resume compares.
                recording_details: dict[str, object] = (
                    {}
                    if collection.pinned_opponent is None
                    else {"pinned_opponent": collection.pinned_opponent}
                )
                writer = RunWriter(
                    root / "episodes",
                    phase="training",
                    policies=policies,
                    details=recording_details or None,
                )
                stack.callback(writer.close)
                metadata["recording"] = {
                    "relative_path": writer.run_dir.relative_to(root).as_posix(),
                    "phase": "training",
                    "pass_id": "1",
                    "policies": policies,
                    "checkpoint_id": None,
                    "details": recording_details,
                }
            atomic_json(
                root / "run_details.json",
                {
                    "schema_version": 1,
                    "run_id": metadata["run_id"],
                    "selection_rule": metadata["selection_rule"],
                    "created_at": started_at,
                    "initial_setup_seconds": time.monotonic() - attempt_started,
                    "config": metadata["config"],
                    "source": source,
                    "dependencies": dependencies,
                    "schemas": checkpoints.checkpoint_schemas(config.method),
                    "execution": execution_identity_now,
                    "runtime": runtime,
                    "content_binding": collection.binding.model_dump(mode="json"),
                    "schedule": dict(schedule.rounding_report),
                    "panel_digest": None if panel is None else panel.digest,
                    "process": process_identity(),
                    **{
                        name: metadata[name]
                        for name in (
                            "validation_deployment",
                            "validation_declaration",
                            "validation_partner_references",
                        )
                        if name in metadata
                    },
                },
            )
        execution = _Run(
            root,
            config,
            collection,
            state,
            metadata,
            writer,
            panel,
            checkpoint,
            attempt_started=attempt_started,
            runtime=runtime,
            validation_options=validation_options,
            matchmaking=matchmaking,
            on_update=on_update,
        )
        # _Run now owns the learner. Drop these references so the first state
        # (for QMIX, a whole replay) is freed once the run moves past it.
        state = restored = None
        return execution.execute()


def _prepare_extension(
    config: TrainConfig,
    collection: TrainingCollection,
    restored: RestoredCheckpoint,
    panel: FrozenPanel | None,
    request: dict[str, Any],
    source: dict[str, Any],
) -> tuple[TrainConfig, TrainingCollection, Any, FrozenPanel | None, dict[str, Any]]:
    """Check and build a child boundary without writing or taking a real step.

    restored has passed full parent restore and host checks. All numerical
    changes are limited to declared segment accounting and future schedules.
    The result contains checked declarations; the runner publishes them before
    calling the existing execution loop. After all changes, commit the child
    state to the parent's selected device once. This keeps its first and later
    update calls consistent without copying state on every update. Parent files
    remain read only.
    """
    from copy import deepcopy

    import numpy as np

    from marl_battlegrounds.training import validation
    from marl_battlegrounds.training._continuation_schedules import (
        continuation_collection,
        continuation_config,
        continuation_state,
        learner_continuation,
        resolve_learner_continuation,
    )
    from marl_battlegrounds.training._selection_evidence import (
        freeze_inherited_candidates,
    )
    from marl_battlegrounds.training.collection import (
        _begin_training_segment,  # pyright: ignore[reportPrivateUsage]
        _continuation_history_thresholds,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training.curriculum import (
        _continuation_details,  # pyright: ignore[reportPrivateUsage]
        _make_continuation_schedule,  # pyright: ignore[reportPrivateUsage]
        _restore_continuation_schedule,  # pyright: ignore[reportPrivateUsage]
    )

    state = restored.state
    saved = restored.details
    parent = saved["metadata"]
    prior = parent.get("continuation")
    start = saved["counters"]["env_steps"]
    total = start + request["additional_env_steps"]
    changes = request["changes"]
    history = state.carry.history
    from marl_battlegrounds.training.shaping import resolve_reward

    child_reward, child_reward_identity = (
        resolve_reward(changes["reward"])
        if "reward" in changes
        else (collection.reward, collection.reward_identity)
    )
    if child_reward_identity is not None and collection.reward_identity is not None:
        original_definition = {
            key: value
            for key, value in collection.reward_identity.items()
            if key not in ("component_schema", "component_keys")
        }
        if child_reward_identity == original_definition:
            child_reward_identity = collection.reward_identity
    changed_custom_reward = child_reward_identity != collection.reward_identity
    settings_before = continuation_config(
        config.qmix or config.pqn or config.ppo,
        learner_continuation(None if prior is None else prior["learner"]),
    )
    family = method_settings_field(config.method)
    gamma_after = changes.get(family, {}).get("gamma", settings_before.gamma)
    shaping_after = changes.get("shaping", config.shaping)
    mode_after = changes.get("shaping_mode", config.shaping_mode)
    changed_reward = (
        changed_custom_reward
        or (
            shaping_after,
            mode_after,
            changes.get("shaping_coefficient", config.shaping_coefficient),
        )
        != (config.shaping, config.shaping_mode, config.shaping_coefficient)
        or (
            gamma_after != settings_before.gamma
            and (
                (config.shaping and config.shaping_mode == "potential")
                or (shaping_after and mode_after == "potential")
            )
        )
    )
    reward_reset = changed_reward and not is_ppo_method(config.method)
    frozen_epsilon = (
        ()
        if is_ppo_method(config.method)
        else tuple(
            float(value)
            for value in np.asarray(history.historical_variables.epsilon)[
                np.asarray(history.captured_ids) >= 0
            ]
        )
    )
    learner_context = resolve_learner_continuation(
        method=config.method,
        settings=config_to_dict(config)[method_settings_field(config.method)],
        counters=saved["counters"],
        num_envs=config.num_envs,
        total_env_steps=total,
        changes=changes,
        parent=None if prior is None else prior["learner"],
        frozen_epsilon=frozen_epsilon,
        frozen_capture_ids=()
        if is_ppo_method(config.method)
        else tuple(
            int(value) for value in np.asarray(history.captured_ids) if value >= 0
        ),
        pinned_epsilon=None
        if is_ppo_method(config.method)
        or history.pinned_variables is None
        or int(history.pinned_update) < 0
        else float(history.pinned_variables.epsilon),
        original_total_env_steps=config.total_env_steps,
        reward_reset=reward_reset,
        last_refresh_rounds=int(history.last_refresh_rounds) if reward_reset else None,
    )
    parent_validation = validation.saved_validation_declaration(
        parent, panel
    ) or validation.run_validation_declaration(config_to_dict(config), panel)
    validation_changes: Any = changes.get("validation", {})
    if not isinstance(validation_changes, Mapping):
        raise ValueError("validation changes must be a mapping")
    validation_changes = cast(dict[str, Any], validation_changes)
    if "panel" in validation_changes:
        panel = validation.load_panel(
            Path(validation_changes["panel"]).resolve(),
            bindings=request.get("validation_bindings"),
            red_zone_depth=config.red_zone_depth,
        )
    root_total = (
        config.total_env_steps
        if prior is None
        else prior["segment"]["root_schedule"]["total_env_steps"]
    )
    config = _child_config(
        {
            **config_to_dict(config),
            "validation_panel": None if panel is None else str(panel.path.absolute()),
            "validation_opponents": config.validation_opponents
            if panel is None
            else None,
            "checkpoint_env_steps": [
                value for value in config.checkpoint_env_steps if start < value <= total
            ],
            "opponents": changes.get("opponents", config.opponents),
            "partners": changes.get("partners", config.partners),
            **{
                name: changes.get(name, getattr(config, name))
                for name in ("shaping", "shaping_mode", "shaping_coefficient")
            },
            "reward": changes.get("reward", config.reward),
            "retain_all_actors": changes.get(
                "retain_all_actors", config.retain_all_actors
            ),
            "keep_past": changes.get("keep_past", config.keep_past),
            "past_capture_interval": changes.get(
                "past_capture_interval",
                None
                if "history_capture_env_steps" in changes
                else config.past_capture_interval,
            ),
            "history_capture_env_steps": changes.get(
                "history_capture_env_steps",
                None
                if "past_capture_interval" in changes
                else config.history_capture_env_steps,
            ),
        },
        total_env_steps=total,
        root_total_env_steps=root_total,
    )
    if "panel" in validation_changes:
        _check_panel_purpose(config, panel)
    context: dict[str, Any] = {
        "schema_version": 1,
        "parent_checkpoint": str(restored.path),
        "parent_checkpoint_id": saved["checkpoint_id"],
        "parent_run_id": parent["run_id"],
        "root_run_id": parent["run_id"] if prior is None else prior["root_run_id"],
        "parent_source": parent["source"],
        "child_source": source,
        "start_env_steps": start,
        "additional_env_steps": request["additional_env_steps"],
        "resulting_total_env_steps": total,
        "changes": changes,
        "learner": learner_context,
        "parent_validation_declaration": parent_validation,
        "parent_statistics": deepcopy(parent["host_state"]),
        "inherited_candidates": [],
    }
    if changed_custom_reward and collection.reward_component_keys:
        from marl_battlegrounds.training.shaping import (
            _merge_reward_records,  # pyright: ignore[reportPrivateUsage]
            _reward_totals_record,  # pyright: ignore[reportPrivateUsage]
            _RewardTotals,  # pyright: ignore[reportPrivateUsage]
        )

        assert isinstance(state.carry.reward_totals, _RewardTotals)
        context["parent_statistics"]["collected_reward_components"] = (
            _merge_reward_records(
                context["parent_statistics"].get("collected_reward_components"),
                _reward_totals_record(
                    state.carry.reward_totals, collection.reward_component_keys
                ),
            )
        )
    declared = validation.run_validation_declaration(
        config_to_dict(config), panel, continuation=context
    )
    context["inherited_candidates"] = freeze_inherited_candidates(
        restored.path, declaration=declared
    )
    thresholds = None
    if "history_capture_env_steps" in changes:
        wanted_raw = changes["history_capture_env_steps"]
        if not isinstance(wanted_raw, list):
            raise ValueError("history_capture_env_steps must be a list")
        wanted = cast(list[object], wanted_raw)
        if any(type(value) is not int for value in wanted):
            raise ValueError("history_capture_env_steps must contain whole numbers")
        steps = cast(list[int], wanted)
        if any(
            not start < value <= total or value % config.num_envs for value in steps
        ):
            raise ValueError(
                "History targets must follow the saved step, fit the child end, "
                "and contain whole environment rounds"
            )
        thresholds = _continuation_history_thresholds(
            state.carry,
            future_rounds=tuple(value // config.num_envs for value in steps),
        )
    schedule = _make_continuation_schedule(
        collection.schedule,
        completed_rounds=start // config.num_envs,
        additional_env_steps=request["additional_env_steps"],
        history_threshold_rounds=thresholds,
        curriculum=changes.get("curriculum"),
    )
    _check_happo_rosters(config.method, config.ppo, changes.get("curriculum"))
    collection, carry = _begin_training_segment(
        collection, state.carry, schedule=schedule
    )
    segment = _continuation_details(collection.schedule)
    assert segment is not None
    segment["learner"] = learner_context
    schedule = _restore_continuation_schedule(segment)
    collection = replace(collection, schedule=schedule)
    rules = learner_continuation(learner_context)
    collection = continuation_collection(collection, rules)
    import jax
    import jax.numpy as jnp

    from marl_battlegrounds.training._continuation_schedules import (
        continuation_boundary,
    )
    from marl_battlegrounds.training.collection import (
        _resize_training_history,  # pyright: ignore[reportPrivateUsage]
    )

    interval = -(-(config.past_capture_interval or 0) // config.num_envs)
    minimum = int(carry.history.minimum_capture_rounds)
    if config.keep_past and interval and interval < minimum:
        raise ValueError(f"Past captures need at least {minimum} rounds between copies")
    if interval or not config.keep_past:
        # Future recurring requests use the saved actual capture clock.
        schedule = replace(
            schedule,
            arrays=schedule.arrays._replace(
                history_threshold_rounds=jnp.zeros(20, jnp.int32),
                history_threshold_count=jnp.int32(0),
            ),
        )
        # Keep the saved segment declaration consistent with its numerical table.
        segment["history_threshold_rounds"] = (0,) * 20
        segment["history_threshold_count"] = 0
        schedule = _restore_continuation_schedule(segment)
        carry = carry._replace(
            schedule=schedule.arrays,
            history=carry.history._replace(
                threshold_to_snapshot=jnp.full(20, -1, jnp.int32)
            ),
        )
        collection = replace(collection, schedule=schedule)
    active_count = (
        20
        if carry.schedule.history_threshold_count is None
        else int(carry.schedule.history_threshold_count)
    )
    pending = [
        int(point)
        for point, identity in zip(
            np.asarray(carry.schedule.history_threshold_rounds)[:active_count],
            np.asarray(carry.history.threshold_to_snapshot)[:active_count],
            strict=True,
        )
        if identity < 0 and point <= total // config.num_envs
    ]
    last_capture = int(carry.history.last_capture_rounds)
    assert rules is not None
    settings = config.qmix or config.pqn or config.ppo
    if config.keep_past and not interval:
        previous_capture = None if last_capture < 0 else last_capture
        for point in pending:
            actual = continuation_boundary(
                max(start // config.num_envs + 1, point),
                total_rounds=total // config.num_envs,
                continuation=rules,
                rollout_length=settings.rollout_length,
                initial_rounds=0 if config.pqn is None else config.pqn.initial_rounds,
            )[0]
            if previous_capture is not None and actual - previous_capture < minimum:
                raise ValueError(
                    f"Continued history captures would be "
                    f"{actual - previous_capture} rounds apart; need at least {minimum}"
                )
            previous_capture = actual
    future = (
        0
        if not config.keep_past
        else -(-request["additional_env_steps"] // config.num_envs // interval) + 1
        if interval
        else len(pending)
    )
    capacity = max(
        int(carry.history.capture_capacity),
        int(carry.history.next_capture_id) + future,
    )
    collection, carry = _resize_training_history(
        collection, carry, keep_past=config.keep_past, history_capture_capacity=capacity
    )
    carry = carry._replace(
        history=carry.history._replace(capture_interval_rounds=jnp.int32(interval))
    )
    collection = replace(collection, carry_spec=jax.eval_shape(_identity, carry))
    state = state._replace(carry=carry)
    if "exploration" in changes:
        state = continuation_state(
            state,
            rules,
            num_envs=config.num_envs,
            initial_rounds=0 if config.pqn is None else config.pqn.initial_rounds,
        )
    if "seed" in changes:
        import jax

        from marl_battlegrounds.training._continuation_schedules import (
            branch_learner_keys,
        )

        state = branch_learner_keys(state, changes["seed"], method=config.method)
        collection = replace(
            collection,
            root_bits=tuple(
                int(x) for x in np.asarray(jax.random.key_data(state.carry.root_key))
            ),
        )
    if changed_reward or gamma_after != settings_before.gamma or "reward" in changes:
        from marl_battlegrounds.training.collection import change_training_reward

        settings_after = continuation_config(
            config.qmix or config.pqn or config.ppo, rules
        )
        collection, carry = change_training_reward(
            collection,
            state.carry,
            reward=child_reward,
            reward_identity=child_reward_identity,
            shaping=config.shaping,
            coefficient=config.shaping_coefficient,
            shaping_mode=config.shaping_mode,
            discount=settings_after.gamma,
        )
        state = state._replace(carry=carry)
        if reward_reset:
            if config.qmix is not None:
                from marl_battlegrounds.training.qmix_learner import (
                    _init_replay,  # pyright: ignore[reportPrivateUsage]
                )

                state = state._replace(
                    replay=_init_replay(collection, carry, config.qmix)
                )
            elif config.pqn is not None:
                from marl_battlegrounds.training.pqn_learner import (
                    _empty_recent,  # pyright: ignore[reportPrivateUsage]
                )

                state = state._replace(
                    recent=_empty_recent(collection, carry, config.pqn)
                )
    context["reward_change"] = {
        "changed": changed_reward,
        "changed_custom_reward": changed_custom_reward,
        "cleared_stored_experience": reward_reset,
        "optimizer_and_statistics": "carried_over",
    }
    context["key_root_bits"] = list(collection.root_bits)
    context["segment"] = _continuation_details(schedule)
    from marl_battlegrounds.training._compilation import (
        _place_training_state,  # pyright: ignore[reportPrivateUsage]
    )

    # Later changes can add fresh arrays after the carry was placed.
    state = _place_training_state(
        state, sharding=restored.state.carry.root_key.sharding
    )
    return (
        config,
        collection,
        state,
        panel,
        {
            "continuation": context,
            "validation_declaration": declared,
            "training_lineage": {
                "root_run_id": context["root_run_id"],
                "parent_run_id": parent["run_id"],
            },
        },
    )


def _training_registrations(collection: TrainingCollection) -> dict[str, object]:
    """Describe the deployed teams once for writer setup and fork preflight."""
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
    )

    return {
        name: normalize_system_registration(system, phase="training")[1]
        for name, system in (
            ("team_a", collection.actor),
            ("team_b", collection.opponent),
        )
    }


def _prepare_extension_recording(
    restored: RestoredCheckpoint,
    state: Any,  # noqa: ANN401
    collection: TrainingCollection,
) -> PreparedRecordingFork:
    """Check a parent's full recording boundary and retain only unfinished games.

    restored is the checked full parent checkpoint; state is its prepared child
    state before another action. This reads the parent's files and uses owned
    temporary streams but creates no child path. The caller must close the
    returned preparation, including when child folder or lock setup fails.
    """
    import numpy as np

    from marl_battlegrounds.evaluation._recording.recording_checkpoint import (
        prepare_recording_fork,
    )

    record = restored.details["metadata"]["recording"]
    mask = np.asarray(
        ~state.carry.state.done.done & state.carry.tracking.first_transition_seen
    )
    episodes = np.asarray(state.carry.state.episode_id)[mask].tolist()
    return prepare_recording_fork(
        restored.path.parent.parent / record["relative_path"],
        restored.details["recording_token"],
        episode_ids=episodes,
        policies=_training_registrations(collection),
    )


def _resize_member_statistics(
    host: dict[str, Any], *, old_capacity: int, new_capacity: int, new_rows: int
) -> None:
    """Keep stable member totals when a child reserves more captures or members.

    Rows are self, legacy pin, capture IDs, then named members. New capture
    rows go before existing named members; newly declared members append.
    Existing totals remain exact Python integers. This changes only the child
    host copy, never the parent checkpoint or running games.
    """
    inserted = new_capacity - old_capacity
    if inserted < 0:
        raise ValueError("A child cannot drop historical result rows")
    for key, width in (
        ("member_completed", 3),
        ("member_score_sums", 2),
        ("last_member_completed", 3),
        ("last_member_score_sums", 2),
    ):
        if key not in host:
            continue
        rows = host[key]
        rows[old_capacity + 2 : old_capacity + 2] = [
            [0] * width for _ in range(inserted)
        ]
        rows.extend([[0] * width for _ in range(new_rows - len(rows))])
        if len(rows) != new_rows:
            raise ValueError("Child member-result rows differ from its population")
    for key in ("sampled_exposure", "used_exposure"):
        if key in host:
            counts = host[key]["by_opponent"]
            counts[old_capacity + 2 : old_capacity + 2] = [0] * inserted
            counts.extend([0] * (new_rows - len(counts)))
            if len(counts) != new_rows:
                raise ValueError("Child exposure rows differ from its population")
    for key in ("member_length_sum", "last_member_length_sum"):
        counts = host.get(key)
        if counts is not None:
            counts[old_capacity + 2 : old_capacity + 2] = [0] * inserted
            counts.extend([0] * (new_rows - len(counts)))
            if len(counts) != new_rows:
                raise ValueError("Child member length rows differ from its population")
    if "ppo_exposure" in host:
        for kind in ("policy", "value"):
            counts = host["ppo_exposure"][f"{kind}_opponent"]
            counts[old_capacity + 2 : old_capacity + 2] = [0] * inserted
            counts.extend([0] * (new_rows - len(counts)))
            if len(counts) != new_rows:
                raise ValueError("Child PPO exposure rows differ from its population")


def _start_extension(
    *,
    root: Path,
    config: TrainConfig,
    collection: TrainingCollection,
    state: Any,  # noqa: ANN401
    panel: FrozenPanel | None,
    restored: RestoredCheckpoint,
    declarations: dict[str, Any],
    source: dict[str, Any],
    dependencies: object,
    execution: dict[str, Any],
    runtime: dict[str, Any],
    started_at: str,
    attempt_started: float,
    stack: ExitStack,
    prepared_recording: PreparedRecordingFork | None = None,
    validation_options: dict[str, Any] | None = None,
    matchmaking: Callable[[dict[str, Any], int], Mapping[str, object]] | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> _Run:
    """Publish one checked child setup and enter the shared execution owner.

    All declarations, parent numerical state and recording were checked before
    creating root. The caller holds root's run lock and has checked that root
    holds nothing but its lock file. prepared_recording owns checked unfinished
    replay streams; stack owns it and any child recording writer. This never
    opens the parent writer for mutation or copies its completed replay/log
    history.
    """
    from copy import deepcopy

    from marl_battlegrounds.training import checkpoints
    from marl_battlegrounds.training.callbacks import (
        _callback_identities,  # pyright: ignore[reportPrivateUsage]
    )

    host = deepcopy(restored.details["metadata"]["host_state"])
    if declarations["continuation"]["reward_change"]["changed_custom_reward"]:
        host.pop("collected_reward_components", None)
    if config.retain_all_actors and not restored.details["metadata"]["config"].get(
        "retain_all_actors", False
    ):
        host["retention_start_capture_id"] = int(state.carry.history.next_capture_id)
    host.update(
        routine_results=[],
        confirmation_results=[],
        random_results=[],
        validation_games=0,
        validation_seconds=0.0,
        actors={},
        pending=None,
        selected_actor=None,
        final_actor=None,
        slot_complete=False,
        selection=None,
        recovery_checkpoints=[],
        exposure_snapshots={},
    )
    if "ppo_exposure" in host:
        for kind in ("policy", "value"):
            counts = host["ppo_exposure"][f"{kind}_source"]
            counts.extend(
                [0] * (len(collection.binding.source_configurations) - len(counts))
            )
    if config.qmix is not None or config.pqn is not None:
        from marl_battlegrounds.training.qmix_learner import (
            _bank_size,  # pyright: ignore[reportPrivateUsage]
        )

        key = "sampled_exposure" if config.qmix is not None else "used_exposure"
        counts = host[key]["by_source"]
        counts.extend([0] * (_bank_size(state.carry) - len(counts)))
    _resize_member_statistics(
        host,
        old_capacity=int(restored.state.carry.history.capture_capacity),
        new_capacity=int(state.carry.history.capture_capacity),
        new_rows=state.carry.progress.opponent_steps.shape[0],
    )
    # These totals use each game's original reset-time stage, like cumulative
    # exposure. Keep them across child accounting segments and add new finishes.
    metadata: dict[str, Any] = {
        "run_id": uuid.uuid4().hex,
        "attempt_id": uuid.uuid4().hex,
        "parent_checkpoint": None,
        "selection_rule": "point_margin",
        "config": config_to_dict(config),
        "source": source,
        "dependencies": dependencies,
        "execution": execution,
        "host_state": host,
        "recording": None,
        "initial_actor": restored.details["metadata"].get("initial_actor"),
        "callback_inputs": _callback_identities(
            on_update=on_update, matchmaking=matchmaking
        ),
        **declarations,
    }
    writer = None
    if config.recording:
        from marl_battlegrounds.evaluation._recording.recording_checkpoint import (
            attach_recording_fork,
        )
        from marl_battlegrounds.evaluation.run_writer import RunWriter

        if prepared_recording is None:
            raise ValueError("Child recording must be checked before creating output")
        record = restored.details["metadata"]["recording"]
        policies = _training_registrations(collection)
        recording_details: dict[str, object] = (
            {}
            if collection.pinned_opponent is None
            else {"pinned_opponent": collection.pinned_opponent}
        )
        writer = RunWriter(
            root / "episodes",
            phase="training",
            pass_id=record["pass_id"],
            policies=policies,
            checkpoint_id=record["checkpoint_id"],
            details=recording_details or None,
        )
        stack.callback(writer.close)
        metadata["continuation"]["recording"] = attach_recording_fork(
            writer, prepared_recording
        )
        metadata["recording"] = {
            **record,
            "policies": policies,
            "details": recording_details,
            "relative_path": writer.run_dir.relative_to(root).as_posix(),
        }
    atomic_json(
        root / "run_details.json",
        {
            "schema_version": 1,
            "run_id": metadata["run_id"],
            "selection_rule": metadata["selection_rule"],
            "created_at": started_at,
            "initial_setup_seconds": time.monotonic() - attempt_started,
            "config": metadata["config"],
            "source": source,
            "dependencies": dependencies,
            "schemas": checkpoints.checkpoint_schemas(config.method),
            "execution": execution,
            "runtime": runtime,
            "content_binding": collection.binding.model_dump(mode="json"),
            "schedule": dict(collection.schedule.rounding_report),
            "panel_digest": None if panel is None else panel.digest,
            "process": process_identity(),
            **declarations,
        },
    )
    return _Run(
        root,
        config,
        collection,
        state,
        metadata,
        writer,
        panel,
        None,
        attempt_started=attempt_started,
        runtime=runtime,
        validation_options=validation_options,
        on_update=on_update,
        matchmaking=matchmaking,
    )


def _crosses_interval(before: int, after: int, interval: int) -> bool:
    """Say whether an update count crossed a multiple of the save interval.

    before and after are completed_updates around one block (PPO updates, or
    QMIX and PQN-VDN optimizer steps); interval is the positive save interval.
    PPO adds one update per block, so this is its historical
    ``after % interval == 0`` rule. A QMIX block adds epochs steps and a
    PQN-VDN learning block adds epochs times num_minibatches steps (an initial
    PQN-VDN chunk adds none), so 24 to 28 crosses 25 while 20 to 24 does not.

    Examples
    --------
    >>> _crosses_interval(24, 28, 25), _crosses_interval(20, 24, 25)
    (True, False)
    """
    return after // interval != before // interval


def _identity(value: Any) -> Any:  # noqa: ANN401
    """Return value unchanged; traced by jax.eval_shape for shape-only copies."""
    return value


def _runtime_details(package_version: str, num_envs: int) -> dict[str, Any]:
    """Read evaluation runtime facts and declared device selectors once per attempt.

    package_version is the verified source version and num_envs is the declared
    positive batch. Reuse evaluation's backend/device/precision record after setup has
    selected JAX. Environment values describe selectors, not verified physical
    GPU identities; the frozen launch package owns its UUID/PCI verification.
    No model call, timing, transfer, file write or driver query occurs here.
    """
    from marl_battlegrounds.evaluation.runtime_provenance import (
        capture_runtime_provenance,
    )

    return {
        "provenance": capture_runtime_provenance(
            package_version, num_envs=num_envs, policy_execution_included=True
        ).model_dump(mode="json"),
        "declared_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "JAX_DISABLE_JIT",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
            )
        },
    }


def _memory_snapshot() -> dict[str, Any]:
    """Read process peak RAM and JAX allocator counters after an attempt ends.

    Return optional measurements and explicit read errors. RAM is the process
    lifetime high-water mark, including earlier work in a reused Python process.
    JAX counters exclude driver/display memory and CPU worker processes. Missing
    allocator statistics are None. This performs no GPU synchronization, array
    transfer, model call, polling or file write. A failed measurement cannot hide
    the attempt's original failure. Call only after the learner loop/report work.
    """
    result: dict[str, Any] = {
        "process_peak_ram_bytes": None,
        "jax_allocator": None,
        "errors": [],
        "scope": (
            "Process lifetime, including earlier work and compilation. JAX "
            "allocator counters exclude driver/display memory and CPU workers."
        ),
    }
    try:
        import resource
        import sys

        scale = 1 if sys.platform == "darwin" else 1024
        result["process_peak_ram_bytes"] = (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale
        )
    except Exception as error:
        result["errors"].append(f"Process RAM: {type(error).__name__}: {error}")
    try:
        import jax

        result["jax_allocator"] = jax.devices()[0].memory_stats()
    except Exception as error:
        result["errors"].append(f"JAX allocator: {type(error).__name__}: {error}")
    return result


def _random_progress(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarize already verified Random games for the human progress display.

    results contains this run's completed Random captures, including an optional
    shared initialization. Return the latest capture's game counts, equal-map
    mean kills and deaths, and its change from initialization. Kills are read
    from the columns analysis._kill_columns names for each record: recorded
    kills for records that carry a Red Zone depth (a Red Zone death gives 2
    points but is still one kill), the score columns for older records, where
    points and kills were equal. Missing combat values remain None. Capture
    times are the times when the actor was saved, not the time when its
    evaluation finished. Empty results return None. This only reads small host
    dictionaries; it opens no files, touches no device arrays and changes
    neither evidence nor checkpoint state.
    """
    from marl_battlegrounds.training.analysis import (
        _kill_columns,  # pyright: ignore[reportPrivateUsage]
    )

    if not results:
        return None

    def mean_scores(record: dict[str, Any]) -> tuple[float | None, float | None]:
        """Average map kills for and against equally; missing values stay None."""
        cells = record.get("cells", [])
        if not cells:
            return None, None
        team_a, team_b = (f"mean_{name}" for name in _kill_columns(record))
        first = [cell.get(team_a) for cell in cells]
        second = [cell.get(team_b) for cell in cells]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in (*first, *second)
        ):
            return None, None
        return math.fsum(first) / len(cells), math.fsum(second) / len(cells)

    latest = max(results, key=lambda record: record["env_steps"])
    initial = next((record for record in results if record["env_steps"] == 0), None)
    kills, deaths = mean_scores(latest)
    margin = None if kills is None or deaths is None else kills - deaths
    initial_margin = None
    if initial is not None:
        initial_kills, initial_deaths = mean_scores(initial)
        if initial_kills is not None and initial_deaths is not None:
            initial_margin = initial_kills - initial_deaths
    return {
        "opponent": "Random",
        "task_id": latest["task_id"],
        "initial_task_id": None if initial is None else initial["task_id"],
        "env_steps": latest["env_steps"],
        "games": latest["games"],
        "wall_seconds": latest["wall_seconds"],
        "training_seconds": latest["training_seconds"],
        "score": latest["score"],
        **{
            name: sum(cell[name] for cell in latest["cells"])
            for name in ("wins", "draws", "losses")
        },
        "mean_kills_for": kills,
        "mean_kills_against": deaths,
        "kill_margin": margin,
        "initial_kill_margin": initial_margin,
        "kill_margin_change": None
        if margin is None or initial_margin is None
        else margin - initial_margin,
    }


def _finite_or_marked(value: object, name: str, markers: dict[str, str]) -> object:
    """Copy a float or nested list of floats, replacing nonfinite values by None.

    Parameters
    ----------
    value : object
        A Python float, bool or int, or a nested list of them (for example a
        rejected block's (epochs, minibatches) losses after ``tolist()``).
    name : str
        Field name used in the markers, such as ``"loss"``.
    markers : dict[str, str]
        Mapping this function adds to: one entry per replaced value, keyed
        by the name plus its indices, such as ``"loss[1,0]"``, with the value
        "NaN", "+Inf" or "-Inf". Existing entries are kept.

    Returns
    -------
    object
        The same structure with every nonfinite float replaced by None.
        Finite floats, booleans and integers are returned unchanged; an empty
        list stays empty.

    Notes
    -----
    Host-only and pure apart from adding to markers. The shared JSON writer
    refuses NaN, so the event keeps every value without disguising it.
    """

    def clean(item: object, index: tuple[int, ...]) -> object:
        """Replace one nonfinite float in place of its position."""
        if isinstance(item, list):
            return [
                clean(entry, (*index, position))
                for position, entry in enumerate(cast(list[object], item))
            ]
        if isinstance(item, float) and not math.isfinite(item):
            label = name if not index else f"{name}[{','.join(map(str, index))}]"
            markers[label] = (
                "NaN" if math.isnan(item) else ("+Inf" if item > 0 else "-Inf")
            )
            return None
        return item

    return clean(value, ())


def _rejected_block_facts(result: Any) -> dict[str, object]:  # noqa: ANN401
    """Describe a rejected PQN-VDN block for the learner_update_rejected event.

    Parameters
    ----------
    result : PQNUpdateResult
        Host copy of the rejected block's result.

    Returns
    -------
    dict
        reason (int32 failure code), generated_transitions and
        active_samples from the kept summary, task_reward_sum and
        shaping_reward_sum, and the attempted (epochs, minibatches) step
        flags (steps_performed, steps_finite) and losses. Finite values are
        written as they are; each nonfinite value is None and the
        "nonfinite" object names it with its marker.

    Notes
    -----
    Host-only; the runner writes the event, then attempt_failed, then raises.
    No retry happens and no host count changes.
    """
    import numpy as np

    summary, metrics = result.summary, result.metrics
    markers: dict[str, str] = {}
    facts: dict[str, object] = {
        "reason": int(result.failure_reason),
        "generated_transitions": int(summary.real_transitions),
        "active_samples": int(summary.active_samples),
        "task_reward_sum": _finite_or_marked(
            float(summary.task_reward_sum), "task_reward_sum", markers
        ),
        "shaping_reward_sum": _finite_or_marked(
            float(summary.shaping_reward_sum), "shaping_reward_sum", markers
        ),
        "steps_performed": np.asarray(metrics.performed).tolist(),
        "steps_finite": np.asarray(metrics.finite).tolist(),
        "losses": _finite_or_marked(
            np.asarray(metrics.loss, dtype=np.float64).tolist(), "losses", markers
        ),
    }
    facts["nonfinite"] = markers
    return facts


def _gradient_norm_summary(
    norms: object, performed: object
) -> dict[str, float | int | None]:
    """Summarize one block's gradient norms for a QMIX or PQN-VDN log row.

    Parameters
    ----------
    norms : array-like
        Host copy of the block's per-step global gradient norms, float32, with
        shape (epochs,) for QMIX or (epochs, minibatches) for PQN-VDN. Values
        may be +inf or NaN when the squared gradients overflowed float32.
    performed : array-like or bool
        True where an optimizer step actually ran, with the same shape as
        norms or a scalar that broadcasts to it. Steps that did not run (no
        eligible pair, or a block without learning) are left out, because
        their stored norm is a placeholder 0.

    Returns
    -------
    dict
        Three JSON-safe keys:
        ``grad_norm_mean`` is the mean of the finite norms of the steps that
        ran, as a Python float; ``grad_norm_max`` is their maximum; both are
        None when no such finite norm exists (for example a warm-up block or
        an initial chunk). ``grad_norm_nonfinite`` is the int count of steps
        that ran with a +inf, -inf or NaN norm; it is 0 when every norm is
        finite or no step ran.

    Raises
    ------
    ValueError
        performed cannot broadcast to the shape of norms.

    Notes
    -----
    Host-only and pure; reads values that the block result already copied to
    the host, so it adds no device work. The values are never inf or NaN, so
    the row passes the shared JSON writer, which refuses them.
    """
    import numpy as np

    values = np.asarray(norms, dtype=np.float64)
    ran = np.broadcast_to(np.asarray(performed, dtype=np.bool_), values.shape)
    used = values[ran]
    finite = used[np.isfinite(used)]
    return {
        "grad_norm_mean": float(np.mean(finite)) if finite.size else None,
        "grad_norm_max": float(np.max(finite)) if finite.size else None,
        "grad_norm_nonfinite": int(used.size - finite.size),
    }


def _checked_export_provenance(
    saved: dict[str, Any], requested: dict[str, object]
) -> dict[str, Any]:
    """Reuse original actor provenance when only checked input paths have moved.

    saved is an existing export's description metadata; requested is the current
    boundary or capture declaration. Compare scientific facts exactly. Ignore
    only display names and use the existing member-binding checker for relocated
    identical opponent and partner inputs. Return the original saved metadata;
    export_system still verifies all actor bytes and inference settings. Raise
    ValueError if any other fact differs. No files change here.
    """
    from marl_battlegrounds.training.checkpoints import (
        _checked_member_binding,  # pyright: ignore[reportPrivateUsage]
    )

    previous = {"initial_actor": None, **saved}
    expected = dict(requested)
    previous.pop("display_name", None)
    expected.pop("display_name", None)
    for key in ("opponent_members", "partner_members"):
        old_members = cast(list[dict[str, Any]], previous.get(key, []))
        new_members = cast(list[dict[str, Any]], expected.get(key, []))
        if len(old_members) != len(new_members):
            raise ValueError("Saved actor provenance differs from this boundary")
        for old_member, new_member in zip(old_members, new_members, strict=True):
            _checked_member_binding(old_member, new_member)
        if key in previous:
            expected[key] = previous[key]
    if "pinned_opponent" in previous or "pinned_opponent" in expected:
        _checked_member_binding(
            previous.get("pinned_opponent"), expected.get("pinned_opponent")
        )
        expected["pinned_opponent"] = previous.get("pinned_opponent")
    if previous != expected:
        raise ValueError("Saved actor provenance differs from this boundary")
    return saved


class _Run:
    """Own one locked process attempt and its small host bookkeeping.

    The public train function validates setup/recovery first. This object holds
    one learner state and stable compiled updater; it never stores a rollout
    after its update. Files and validation are performed between numerical calls.
    A QMIX run keeps exact Python-integer block, update and sampled-use totals
    in its host state instead of PPO's policy and value sample counts; a
    PQN-VDN run keeps block, update and used-row totals the same way.
    Checkpoint writes request one host cleanup after an accepted update replaces
    the saved state and drops its temporary arrays.
    """

    def __init__(
        self,
        root: Path,
        config: TrainConfig,
        collection: TrainingCollection,
        state: Any,  # noqa: ANN401
        metadata: dict[str, Any],
        writer: RunWriter | None,
        panel: FrozenPanel | None,
        checkpoint: Path | None,
        *,
        attempt_started: float,
        runtime: dict[str, Any],
        validation_options: dict[str, Any] | None = None,
        matchmaking: Callable[[dict[str, Any], int], Mapping[str, object]]
        | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Attach validated owners and reconstruct host counters from a checkpoint."""
        from functools import partial

        import jax

        from marl_battlegrounds.training._compilation import training_compiler_options
        from marl_battlegrounds.training._continuation_schedules import (
            continuation_config,
            schedule_continuation,
        )
        from marl_battlegrounds.training._selection_evidence import (
            read_inherited_candidates,
        )
        from marl_battlegrounds.training.learner import update_learner
        from marl_battlegrounds.training.validation import (
            resolve_validation_schedule,
            saved_validation_declaration,
        )

        self.validation_options = (
            {} if validation_options is None else validation_options
        )
        self.on_update = on_update
        self.matchmaking = matchmaking
        self.continuation = schedule_continuation(collection.schedule)
        self.validation_declaration = saved_validation_declaration(metadata, panel)
        self.inherited: dict[str, Any] = (
            {"records": [], "actors": {}, "used_roots": []}
            if self.validation_declaration is None
            or metadata.get("continuation") is None
            else read_inherited_candidates(
                metadata["continuation"]["inherited_candidates"],
                declaration=self.validation_declaration,
            )
        )
        self.ppo_reporting = (
            continuation_config(config.ppo, self.continuation)
            if is_ppo_method(config.method)
            else config.ppo
        )
        self.qmix_reporting = (
            None
            if config.qmix is None
            else continuation_config(config.qmix, self.continuation)
        )
        self.qmix = config.qmix
        self.pqn = config.pqn
        self.rollout_length = (
            self.qmix.rollout_length
            if self.qmix is not None
            else self.pqn.rollout_length
            if self.pqn is not None
            else config.ppo.rollout_length
        )
        self.planned_learning_blocks = (
            0
            if self.pqn is None
            else cast(int, self.continuation.pqn_planned_learning_blocks)
            if self.continuation is not None
            else pqn_planned_learning_blocks(
                config.total_env_steps // config.num_envs, self.pqn
            )
        )
        update_context = (
            {} if self.continuation is None else {"continuation": self.continuation}
        )
        self.root, self.config = root, config
        self.collection, self.state = collection, state
        self._checkpoint_cleanup_pending = False
        self.metadata, self.writer, self.panel = metadata, writer, panel
        self.metadata["study_attempt"] = study_attempt_binding(root)
        self.runtime = runtime
        self.host = metadata["host_state"]
        self.host.setdefault(
            "past_copies_start_capture_id",
            int(state.carry.history.next_capture_id)
            if "past_copies" not in self.host
            else 0,
        )
        if config.retain_all_actors:
            self.host.setdefault("retention_start_capture_id", 0)
        defaults: dict[str, Any] = {
            "member_completed": [
                [0, 0, 0] for _ in range(state.carry.progress.opponent_steps.shape[0])
            ],
            "member_score_sums": [
                [0, 0] for _ in range(state.carry.progress.opponent_steps.shape[0])
            ],
            "last_member_completed": [
                [0, 0, 0] for _ in range(state.carry.progress.opponent_steps.shape[0])
            ],
            "last_member_score_sums": [
                [0, 0] for _ in range(state.carry.progress.opponent_steps.shape[0])
            ],
            "past_copies": {},
            "env_steps": 0,
            "completed_updates": 0,
            "actor_decisions": 0,
            "used_policy_samples": 0,
            "used_value_samples": 0,
            "training_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "routine_results": [],
            "confirmation_results": [],
            "random_results": [],
            "actors": {},
            "pending": None,
            "selected_actor": None,
            "final_actor": None,
            "validation_seconds": 0.0,
            "validation_games": 0,
            "save_seconds": 0.0,
            "saves": 0,
            "report_seconds": None,
            "slot_complete": False,
            "selection": None,
            "recovery_checkpoints": [],
        }
        if self.qmix is not None:
            from marl_battlegrounds.training.qmix_learner import (
                _bank_size,  # pyright: ignore[reportPrivateUsage]
            )

            del defaults["used_policy_samples"], defaults["used_value_samples"]
            defaults.update(
                completed_blocks=0,
                learning_blocks=0,
                sampled_sequences=0,
                used_td_pairs=0,
                used_agent_utilities=0,
                sampled_exposure={
                    "by_stage": [0] * 17,
                    "by_source": [0] * _bank_size(state.carry),
                    "by_opponent": [0] * state.carry.progress.opponent_steps.shape[0],
                },
            )
        if self.pqn is not None:
            from marl_battlegrounds.training.pqn_learner import (
                _bank_size as _pqn_bank_size,  # pyright: ignore[reportPrivateUsage]
            )

            del defaults["used_policy_samples"], defaults["used_value_samples"]
            defaults.update(
                completed_blocks=0,
                learning_blocks=0,
                used_sequences=0,
                used_td_pairs=0,
                used_agent_utilities=0,
                used_prefix_td_pairs=0,
                used_exposure={
                    "by_stage": [0] * 17,
                    "by_source": [0] * _pqn_bank_size(state.carry),
                    "by_opponent": [0] * state.carry.progress.opponent_steps.shape[0],
                },
            )
        rows = state.carry.progress.opponent_steps.shape[0]
        # Legacy checkpoints have no measured length totals or PPO grouping.
        # Preserve that absence instead of pretending the earlier games were zero.
        defaults["member_results_start_env_steps"] = (
            0 if "member_completed" in self.host else self.host.get("env_steps", 0)
        )
        defaults["member_length_sum"] = (
            [0] * rows if not self.host.get("env_steps", 0) else None
        )
        defaults["last_member_length_sum"] = (
            [0] * rows if not self.host.get("env_steps", 0) else None
        )
        defaults["exposure_snapshots"] = {}
        if is_ppo_method(config.method):
            defaults["ppo_exposure"] = {
                "start_env_steps": self.host.get("env_steps", 0),
                "policy_offset": self.host.get("used_policy_samples", 0),
                "value_offset": self.host.get("used_value_samples", 0),
                **{
                    f"{kind}_{dimension}": [0] * size
                    for kind in ("policy", "value")
                    for dimension, size in (
                        ("source", len(collection.binding.source_configurations)),
                        ("opponent", rows),
                    )
                },
            }
        if collection.reward_component_keys:
            defaults["collected_reward_components"] = {
                "real_transitions": 0,
                "learner_team_transitions": 0,
                "learner_agent_transitions": 0,
                "components": {
                    name: {"recipient_sum": 0.0, "team_mean_sum": 0.0}
                    for name in collection.reward_component_keys
                },
            }
        for name, default in defaults.items():
            self.host.setdefault(name, default)
        from marl_battlegrounds.training.checkpoints import read_checkpoint_description

        self.checkpoint = checkpoint
        self.checkpoint_id = (
            None
            if checkpoint is None
            else read_checkpoint_description(checkpoint)["checkpoint_id"]
        )
        if (
            self.checkpoint_id is not None
            and self.checkpoint_id not in self.host["recovery_checkpoints"]
        ):
            self.host["recovery_checkpoints"].append(self.checkpoint_id)
        self.start = attempt_started
        self.prior_elapsed = self.host["elapsed_seconds"]
        details = json.loads((self.root / "run_details.json").read_text())
        self.created_at = datetime.fromisoformat(details["created_at"]).timestamp()
        self.reporter = ProgressReporter(enabled=config.verbose)
        self.speed_estimate = TrainingSpeedEstimate() if config.verbose else None
        if self.qmix is not None:
            from marl_battlegrounds.training.qmix_learner import update_qmix_learner

            self.updater = cast(
                "_Updater",
                jax.jit(
                    partial(update_qmix_learner, qmix=self.qmix, **update_context),
                    compiler_options=training_compiler_options(),
                ),
            )
        elif self.pqn is not None:
            from marl_battlegrounds.training.pqn_learner import update_pqn_learner

            # The planned count is a static Python int bound once per run.
            self.updater = cast(
                "_Updater",
                jax.jit(
                    partial(
                        update_pqn_learner,
                        pqn=self.pqn,
                        planned_learning_blocks=self.planned_learning_blocks,
                        **update_context,
                    ),
                    compiler_options=training_compiler_options(),
                ),
            )
        else:
            self.updater = cast(
                "_Updater",
                jax.jit(
                    partial(
                        update_learner,
                        ppo=config.ppo,
                        method=config.method,
                        **update_context,
                    ),
                    compiler_options=training_compiler_options(),
                ),
            )
        self.validation_steps: set[int] = (
            {point["env_steps"] for point in self.validation_declaration["points"]}
            if self.validation_declaration is not None
            else {
                point.env_steps
                for point in resolve_validation_schedule(
                    config.total_env_steps,
                    config.num_envs,
                    rollout_length=self.rollout_length,
                    fractions=config.validation_fractions,
                    initial_rounds=0 if self.pqn is None else self.pqn.initial_rounds,
                )
            }
            if panel is not None
            else set()
        )
        self.random_steps: set[int] = (
            {0, *config.checkpoint_env_steps, config.total_env_steps}
            if config.random_diagnostic_seed_pairs is not None
            else set()
        )
        if self.continuation is not None:
            self.random_steps = {
                value
                for value in self.random_steps
                if value > metadata["continuation"]["start_env_steps"]
            }
        self.status: dict[str, Any] = {}
        if config.random_diagnostic_seed_pairs is not None:
            self.status["random_validation"] = _random_progress(
                self.host["random_results"]
            )

    def event(self, event_name: str, **facts: object) -> None:
        """Durably append one lifecycle event without changing scientific counters."""
        append_jsonl(
            self.root / "run_events.jsonl",
            {
                "event": event_name,
                "time_utc": utc_now(),
                "run_id": self.metadata["run_id"],
                "attempt_id": self.metadata["attempt_id"],
                "study_attempt": self.metadata.get("study_attempt"),
                "env_steps": self.host["env_steps"],
                "training_seconds": self.host["training_seconds"],
                "elapsed_seconds": self.prior_elapsed + time.monotonic() - self.start,
                "wall_seconds": max(0.0, time.time() - self.created_at),
                **facts,
            },
            durable=True,
        )

    def set_status(self, phase: str, **facts: object) -> None:
        """Publish host counters and the requested reset-time K without device reads."""
        self.host["elapsed_seconds"] = (
            self.prior_elapsed + time.monotonic() - self.start
        )
        training_seconds = self.host["training_seconds"]
        rate = self.host["env_steps"] / training_seconds if training_seconds else None
        report = self.collection.schedule.rounding_report
        ends = cast(tuple[int, ...], report["cumulative_round_ends"])
        thresholds = cast(
            tuple[int, ...], report.get("score_thresholds", (20,) * len(ends))
        )
        stage = min(
            sum(end * self.config.num_envs <= self.host["env_steps"] for end in ends),
            len(ends) - 1,
        )
        requested_threshold = thresholds[stage]
        self.status.update(
            {
                "schema_version": 1,
                "run_id": self.metadata["run_id"],
                "attempt_id": self.metadata["attempt_id"],
                "study_attempt": self.metadata.get("study_attempt"),
                "training_seconds": self.host["training_seconds"],
                "process": process_identity(),
                "phase": phase,
                "status": "running",
                "total_env_steps": self.config.total_env_steps,
                "env_steps": self.host["env_steps"],
                "score_threshold": requested_threshold,
                "completed_updates": self.host["completed_updates"],
                "final_actor": self.host["final_actor"],
                "selected_actor": self.host["selected_actor"],
                "elapsed_seconds": self.host["elapsed_seconds"],
                "wall_seconds": max(0.0, time.time() - self.created_at),
                "transitions_per_second": rate,
                "estimated_transitions_per_second": None
                if self.speed_estimate is None
                else self.speed_estimate.rate,
                "latest_checkpoint": None
                if self.checkpoint is None
                else str(self.checkpoint),
                "pending_task": self.host["pending"],
                "pending_work_seconds": 0.0
                if phase == "complete"
                else self.pending_seconds()
                if self.config.verbose
                else None,
                "updated_at": utc_now(),
                **facts,
            }
        )
        atomic_json(self.root / "status.json", self.status)
        self.reporter.report(self.status)

    def pending_seconds(self) -> float | None:
        """Estimate output time from completed phase timings, never new GPU work.

        Future confirmation allows its maximum three candidates. Completed
        validation games estimate final-diagnostic cost too; the ETA is an
        approximation, not a promised finish time. Unknown phase costs give None.
        """
        from marl_battlegrounds.training.validation import VALIDATION_MAPS

        if not self.host["saves"] or self.host["report_seconds"] is None:
            return None
        if self.continuation is not None:
            from marl_battlegrounds.training._continuation_schedules import (
                continuation_counts,
            )

            _, learning = continuation_counts(
                self.config.total_env_steps // self.config.num_envs,
                continuation=self.continuation,
                rollout_length=self.rollout_length,
                initial_rounds=0 if self.pqn is None else self.pqn.initial_rounds,
                minimum=0 if self.qmix is None else self.qmix.min_buffer_size,
            )
            if self.pqn is not None:
                remaining_updates = (
                    (learning - self.host["learning_blocks"])
                    * self.pqn.epochs
                    * self.pqn.num_minibatches
                )
            elif self.qmix is not None:
                remaining_updates = (
                    learning - self.host["learning_blocks"]
                ) * self.qmix.epochs
            else:
                remaining_updates = learning - self.host["completed_updates"]
        elif self.pqn is not None:
            remaining_updates = (
                self.planned_learning_blocks - self.host["learning_blocks"]
            ) * (self.pqn.epochs * self.pqn.num_minibatches)
        elif self.qmix is None:
            remaining_updates = math.ceil(
                (self.config.total_env_steps - self.host["env_steps"])
                / (self.config.num_envs * self.config.ppo.rollout_length)
            )
        else:
            _, learning = qmix_fixed_block_counts(
                self.config.total_env_steps // self.config.num_envs,
                rollout_length=self.qmix.rollout_length,
                minimum=self.qmix.min_buffer_size,
            )
            remaining_updates = (
                learning - self.host["learning_blocks"]
            ) * self.qmix.epochs
        interval = cast(int, self.config.checkpoint_interval_updates)
        estimate = (
            math.ceil(remaining_updates / interval)
            * self.host["save_seconds"]
            / self.host["saves"]
            + self.host["report_seconds"]
        )
        random_done = {row["env_steps"] for row in self.host["random_results"]}
        map_count = len(self.validation_options.get("maps", VALIDATION_MAPS))
        partner_count = len(self.validation_options.get("partners", {"learner": None}))
        games_per_pair = map_count * partner_count * 2
        random_games = (
            len(self.random_steps - random_done)
            * games_per_pair
            * (self.config.random_diagnostic_seed_pairs or 0)
        )
        if self.panel is None and not random_games:
            return float(estimate)
        if not self.host["validation_games"]:
            return None
        completed = {row["env_steps"] for row in self.host["routine_results"]}
        remaining_points = len(self.validation_steps - completed)
        games = random_games
        if self.panel is not None:
            games += (
                remaining_points
                * games_per_pair
                * len(self.panel.members)
                * self.config.routine_seed_pairs
            )
            games += (
                max(0, 3 - len(self.host["confirmation_results"]))
                * games_per_pair
                * len(self.panel.members)
                * self.config.confirmation_seed_pairs
            )
        if self.config.slot_diagnostic and not self.host["slot_complete"]:
            games += 3200
        return float(
            estimate
            + games * self.host["validation_seconds"] / self.host["validation_games"]
        )

    def _used_opponent_members(self) -> list[dict[str, Any]]:
        """Return declared frozen members that actually supplied a training action.

        Small existing transition counters establish exposure. Merely declaring
        a zero-share member does not make it a training opponent. Saved records
        keep each member's existing source evidence; this infers no opaque history.
        """
        if not self.collection.opponent_records:
            return []
        import numpy as np

        rows = np.asarray(self.state.carry.progress.opponent_steps)
        offset = int(self.state.carry.history.capture_capacity) + 2
        return [
            record
            for index, record in enumerate(self.collection.opponent_records)
            if np.any(rows[offset + index] > 0)
        ]

    def _used_partner_members(self) -> list[dict[str, Any]]:
        """Retain source evidence only for frozen partners that supplied actions."""
        used = self.state.carry.progress.partner_used
        if used is None:
            return []
        return [
            record
            for record, played in zip(
                self.collection.partner_records, used.tolist(), strict=True
            )
            if played
        ]

    def _exposure_rows(self) -> list[dict[str, Any]]:
        """Read compact marginal exposure with honest missing-data reasons.

        Existing collection counters own played starts/transitions; host update
        summaries own completions and optimizer-used samples. Source and map
        completion cells were not collected and are left unknown. Marginal rows
        never imply joint source/opponent observations. No rollout is transferred.
        """
        from marl_battlegrounds.training.collection import training_summary

        summary = training_summary(self.collection, self.state.carry)
        dimensions = ("stage", "source", "map", "opponent", "opponent_source")
        names = (
            "starts",
            "real_transitions",
            "wins",
            "draws",
            "losses",
            "length_sum",
            "completed_games",
            "used_policy_samples",
            "used_value_samples",
            "used_td_pairs",
        )
        rows: list[dict[str, Any]] = []
        q_exposure = self.host.get("sampled_exposure", self.host.get("used_exposure"))
        ppo_exposure = self.host.get("ppo_exposure")

        def row(
            dimension: str, group: int | str, identity: object, **counts: int | None
        ) -> dict[str, Any]:
            """Build one marginal row; every unavailable count has a reason."""
            value: dict[str, Any] = {
                "dimension": dimension,
                "group_id": str(group),
                "identity": identity,
                **{key: str(group) if key == dimension else None for key in dimensions},
                **{key: counts.get(key) for key in names},
            }
            value["unavailable_reasons"] = {
                key: (
                    "Not applicable to this learner"
                    if (key == "used_td_pairs" and is_ppo_method(self.config.method))
                    or (
                        key in ("used_policy_samples", "used_value_samples")
                        and not is_ppo_method(self.config.method)
                    )
                    else "Not recorded for this dimension or historical prefix"
                )
                for key in names
                if value[key] is None
            }
            if dimension in ("source", "opponent", "stage"):
                index = int(group)
                if q_exposure is not None:
                    value["used_td_pairs"] = q_exposure[f"by_{dimension}"][index]
                    value["unavailable_reasons"].pop("used_td_pairs", None)
                if ppo_exposure is not None and dimension != "stage":
                    for kind in ("policy", "value"):
                        key = f"used_{kind}_samples"
                        if ppo_exposure["start_env_steps"] == 0:
                            value[key] = ppo_exposure[f"{kind}_{dimension}"][index]
                            value["unavailable_reasons"].pop(key, None)
                        else:
                            value["unavailable_reasons"][key] = (
                                "Legacy checkpoint did not record "
                                "the earlier sample groups"
                            )
            rows.append(value)
            return value

        for index, starts in enumerate(
            cast(list[int], summary["starts_by_episode_stage"])
        ):
            outcomes = self.host.get("stage_completed", [[0, 0, 0]] * 17)[index]
            row(
                "stage",
                index,
                {"episode_stage": index},
                starts=starts,
                real_transitions=cast(list[int], summary["steps_by_episode_stage"])[
                    index
                ],
                wins=outcomes[0],
                draws=outcomes[1],
                losses=outcomes[2],
                completed_games=sum(outcomes),
                length_sum=self.host.get("stage_length_sum", [0] * 17)[index],
            )
        for index, identity in enumerate(self.collection.binding.source_configurations):
            row("source", index, {"configuration_digest": identity})
        for index, count in enumerate(cast(list[int], summary["steps_by_map"])):
            row("map", index, {"map_id": index}, real_transitions=count)
        members = self._member_statistics()
        member_rows: list[dict[str, Any]] = []
        for index, member in enumerate(members):
            counts = member["cumulative"]
            member_rows.append(
                row(
                    "opponent",
                    index,
                    {
                        key: value
                        for key, value in member.items()
                        if key not in ("cumulative", "last_update")
                    },
                    starts=cast(list[int], summary["starts_by_opponent"])[index],
                    real_transitions=cast(list[int], summary["steps_by_opponent"])[
                        index
                    ],
                    **{
                        key: counts[key]
                        if self.host.get("member_results_start_env_steps", 0) == 0
                        else None
                        for key in (
                            "wins",
                            "draws",
                            "losses",
                            "completed_games",
                            "length_sum",
                        )
                    },
                )
            )
        for kind, selected_kinds in (
            ("self", {"self"}),
            ("history", {"past"}),
            ("fixed", {"pin", "named"}),
        ):
            selected = [
                value
                for value, member in zip(member_rows, members, strict=True)
                if member["kind"] in selected_kinds
            ]
            combined = row(
                "opponent_source",
                kind,
                {"kind": kind},
                **{
                    name: sum(value[name] for value in selected)
                    if selected and all(value[name] is not None for value in selected)
                    else None
                    for name in names
                },
            )
            for name in names:
                if combined[name] is None:
                    combined["unavailable_reasons"][name] = (
                        "No member rows"
                        if not selected
                        else next(
                            value["unavailable_reasons"][name]
                            for value in selected
                            if value[name] is None
                        )
                    )
        return rows

    def _snapshot_exposure(self) -> None:
        """Append a boundary event before its checkpoint can accept that evidence.

        Only initial, routine validation and final boundaries create snapshots.
        Multiple purposes at one step share an event. Repeated saves reuse the
        same checked ID/digest. A failed save leaves raw unaccepted attempt data;
        only a published checkpoint's mapping makes it active evidence.
        """
        from marl_battlegrounds.training._run_io import exposure_event_digest

        steps = self.host["env_steps"]
        continuation = cast(dict[str, Any], self.metadata.get("continuation") or {})
        start = continuation.get("start_env_steps", 0)
        purposes = ["initial"] if steps == start else []
        if steps in self.validation_steps:
            purposes.append("routine_validation")
        if steps == self.config.total_env_steps:
            purposes.append("final")
        if not purposes:
            return
        boundary = f"{self.metadata['run_id']}:{steps}"
        accepted = self.host["exposure_snapshots"]
        if boundary in accepted:
            return
        requested = [steps]
        if self.validation_declaration is not None:
            requested = next(
                (
                    point["requested_steps"]
                    for point in self.validation_declaration["points"]
                    if point["env_steps"] == steps
                ),
                requested,
            )
        event = {
            "event": "exposure_snapshot",
            "event_id": uuid.uuid4().hex,
            "run_id": self.metadata["run_id"],
            "attempt_id": self.metadata["attempt_id"],
            "boundary_id": boundary,
            "purposes": purposes,
            "time_utc": utc_now(),
            "requested_env_steps": requested,
            "actual_env_steps": steps,
            "training_seconds": self.host["training_seconds"],
            "elapsed_seconds": self.host["elapsed_seconds"],
            "wall_seconds": max(0.0, time.time() - self.created_at),
            "rows": self._exposure_rows(),
        }
        digest = exposure_event_digest(event)
        append_jsonl(self.root / "run_events.jsonl", event, durable=True)
        accepted[boundary] = {"event_id": event["event_id"], "digest": digest}

    def _collect_checkpoint_garbage(self) -> None:
        """Release unused saved-array cycles after dropping a block's old arrays.

        Call only after accepting the new state and deleting block temporaries.
        Several writes share one cleanup; blocks with no pending cleanup skip it. Charge
        the host cost to save_seconds, not collection or learner time. Live
        arrays are retained, and no device work or cache reset is requested.
        """
        if not self._checkpoint_cleanup_pending:
            return
        started = time.monotonic()
        gc.collect()
        self.host["save_seconds"] += time.monotonic() - started
        self._checkpoint_cleanup_pending = False

    def save(self) -> Path:
        """Publish one accepted learner boundary with its exact host-log prefix."""
        from marl_battlegrounds.training.checkpoints import (
            read_checkpoint_description,
            save_checkpoint,
        )

        self.host["elapsed_seconds"] = (
            self.prior_elapsed + time.monotonic() - self.start
        )
        self._snapshot_exposure()
        self.metadata["log_cursors"] = {
            "training_updates.jsonl": log_cursor(self.root / "training_updates.jsonl")
        }
        self.metadata["used_opponent_members"] = self._used_opponent_members()
        if self.collection.partner_records:
            self.metadata["used_partner_members"] = self._used_partner_members()
        if self.state.carry.opponent_selection is not None:
            from marl_battlegrounds.training.collection import (
                _selection_value,  # pyright: ignore[reportPrivateUsage]
            )

            self.metadata["opponent_selection"] = _selection_value(
                self.collection, self.state.carry.opponent_selection
            )
        if self.state.carry.partner_selection is not None:
            from marl_battlegrounds.training.collection import partner_selection_value

            self.metadata["partner_selection"] = partner_selection_value(
                self.collection, self.state.carry
            )
        started = time.monotonic()
        self._checkpoint_cleanup_pending = True
        self.checkpoint = save_checkpoint(
            self.root,
            self.collection,
            self.state,
            metadata=self.metadata,
            writer=self.writer,
            ppo=self.config.ppo,
            method=self.config.method,
            qmix=self.qmix,
            pqn=self.pqn,
        )
        self.checkpoint_id = read_checkpoint_description(self.checkpoint)[
            "checkpoint_id"
        ]
        self.metadata["parent_checkpoint"] = self.checkpoint_id
        self.set_status(self.status.get("phase", "saving"))
        seconds = time.monotonic() - started
        self.host["save_seconds"] += seconds
        self.host["saves"] += 1
        self.event(
            "checkpoint_saved",
            checkpoint_id=self.checkpoint_id,
            seconds=seconds,
        )
        if self.config.retain_all_actors:
            self.actor()
        self.prune_recovery()
        return self.checkpoint

    def prune_recovery(self) -> None:
        """Keep two recent recovery payloads and explicitly used candidate boundaries.

        Only payload directories of older checkpoints from this active attempt's saved
        recovery list are removed after a new durable pointer. Small ancestry
        descriptions, actor exports and all evaluation recording bundles remain.
        """
        from marl_battlegrounds.training.checkpoints import artifact_directory

        assert self.checkpoint is not None and self.checkpoint_id is not None
        recent: list[str] = self.host["recovery_checkpoints"]
        if self.checkpoint_id not in recent:
            recent.append(self.checkpoint_id)
        while len(recent) > 2:
            identifier = recent.pop(0)
            candidates = (
                {
                    row["checkpoint_id"]
                    for key in (
                        "routine_results",
                        "confirmation_results",
                        "random_results",
                    )
                    for row in self.host.get(key, [])
                }
                if self.config.retain_all_actors
                else self.host["actors"]
            )
            if identifier in candidates:
                continue
            if len(identifier) != 64 or any(
                char not in "0123456789abcdef" for char in identifier
            ):
                raise ValueError("Invalid checkpoint retention identity")
            directory = artifact_directory(self.root, "checkpoints", identifier)
            if directory.is_symlink():
                raise ValueError("Checkpoint retention cannot follow a symlink")
            for name in ("actor", "state"):
                payload = directory / name
                if payload.is_symlink():
                    raise ValueError(
                        "Checkpoint retention cannot follow a payload symlink"
                    )
                if payload.exists():
                    shutil.rmtree(payload)
            self.event("checkpoint_payload_pruned", checkpoint_id=identifier)

    def actor(self) -> Path:
        """Export this boundary's exact actor; reuse an identical existing artifact.

        A QMIX export holds only the Q-network parameters and a PQN-VDN export
        only the network's parameters and statistics; both play greedily and
        record the optimizer steps behind them. A checked input artifact may move;
        reusing this actor preserves its original provenance and never rewrites it.
        """
        from marl_battlegrounds.training.checkpoints import (
            artifact_directory,
            artifact_name,
            export_system,
            read_checkpoint_description,
        )

        started = time.monotonic()
        assert self.checkpoint is not None and self.checkpoint_id is not None
        destination = (
            self.root
            / "actors"
            / artifact_name(
                self.config.method, "actor", self.host["env_steps"], self.checkpoint_id
            )
        )
        existing = artifact_directory(self.root, "actors", self.checkpoint_id)
        if existing.exists():
            destination = existing
        destination.parent.mkdir(exist_ok=True)
        variables = self.state.carry.history.current_variables
        settings = self.qmix or self.pqn or self.config.ppo
        provenance: dict[str, object] = {
            "run_id": self.metadata["run_id"],
            "seed": self.config.seed,
            "env_steps": self.host["env_steps"],
            "checkpoint_id": self.checkpoint_id,
            "display_name": (
                f"{self.config.method.replace('_', '-').upper()} at step "
                f"{self.host['env_steps']:,} ({self.root.name.replace('_', ' ')})"
            ),
            **(
                {}
                if is_ppo_method(self.config.method)
                else {"optimizer_steps": self.host["completed_updates"]}
            ),
            "initial_actor": self.metadata.get("initial_actor"),
            **(
                {"opponent_members": used}
                if (used := self._used_opponent_members())
                else {}
            ),
            **(
                {"partner_members": partners}
                if (partners := self._used_partner_members())
                else {}
            ),
            # A later run that pins this export inherits its exposure.
            **(
                {}
                if self.collection.pinned_opponent is None
                or self.collection.opponent_records
                else {"pinned_opponent": self.collection.pinned_opponent}
            ),
        }
        reused = destination.exists()
        if reused:
            provenance = _checked_export_provenance(
                read_checkpoint_description(destination)["metadata"], provenance
            )
        else:
            self._checkpoint_cleanup_pending = True
        export_system(
            variables.params
            if self.qmix is not None
            else variables.network
            if self.pqn is not None
            else variables,
            destination,
            input_scale=settings.input_scale,
            spawn_frame=settings.spawn_frame,
            method=self.config.method,
            parameter_sharing=settings.parameter_sharing,
            normalization="batch_norm" if self.pqn is None else self.pqn.normalization,
            metadata=provenance,
        )
        self.host["actors"][self.checkpoint_id] = str(destination)
        self.event(
            "actor_exported",
            checkpoint_id=self.checkpoint_id,
            path=str(destination),
            reused=reused,
            seconds=time.monotonic() - started,
        )
        return destination

    def _export_past(self, event: Any) -> None:  # noqa: ANN401
        """Export one frozen capture before the next learner update can reuse it.

        The artifact records its capture origin and sampled exploration. It owns
        no recovery payload, so normal checkpoint pruning cannot remove it.
        A resume first sets aside every export from after its restored save
        (``set_aside_abandoned_exports``), so a replayed capture writes a fresh
        export; the existing exporter still verifies every byte and the whole
        declaration before reusing one already in place.
        """
        if not bool(event.created):
            return
        started = time.monotonic()
        import jax

        from marl_battlegrounds.training.checkpoints import (
            export_system,
            read_checkpoint_description,
        )

        capture_id = int(event.capture_id)
        steps = int(event.rounds) * self.config.num_envs

        def captured(value: jax.Array) -> jax.Array:
            """Read this captured slot without retaining any other bank weights."""
            return value[int(event.slot)]

        variables = jax.tree.map(
            captured, self.state.carry.history.historical_variables
        )
        settings = self.qmix or self.pqn or self.config.ppo
        destination = (
            self.root
            / "past_copies"
            / f"{self.config.method}_past_step_{steps}_capture_{capture_id:06d}"
        )
        destination.parent.mkdir(exist_ok=True)
        epsilon = (
            None if is_ppo_method(self.config.method) else float(variables.epsilon)
        )
        provenance: dict[str, object] = {
            "run_id": self.metadata["run_id"],
            "seed": self.config.seed,
            "env_steps": steps,
            "display_name": (
                f"{self.config.method.replace('_', '-').upper()} "
                f"past copy at step {steps:,}"
            ),
            "capture_origin": {
                "capture_id": capture_id,
                "update_index": int(event.update_index),
                "content_binding": self.collection.binding.model_dump(mode="json"),
                "epsilon": epsilon,
            },
            "initial_actor": self.metadata.get("initial_actor"),
            **(
                {"opponent_members": used}
                if (used := self._used_opponent_members())
                else {}
            ),
            **(
                {"partner_members": partners}
                if (partners := self._used_partner_members())
                else {}
            ),
            **(
                {}
                if is_ppo_method(self.config.method)
                else {"optimizer_steps": self.host["completed_updates"]}
            ),
            **(
                {}
                if self.collection.pinned_opponent is None
                or self.collection.opponent_records
                else {"pinned_opponent": self.collection.pinned_opponent}
            ),
        }
        if destination.exists():
            provenance = _checked_export_provenance(
                read_checkpoint_description(destination)["metadata"], provenance
            )
        else:
            self._checkpoint_cleanup_pending = True
        export_system(
            variables.params
            if self.qmix is not None
            else variables.network
            if self.pqn is not None
            else variables,
            destination,
            metadata=provenance,
            method=self.config.method,
            parameter_sharing=settings.parameter_sharing,
            normalization="batch_norm" if self.pqn is None else self.pqn.normalization,
            input_scale=settings.input_scale,
            spawn_frame=settings.spawn_frame,
        )
        self.host["past_copies"][str(capture_id)] = {
            "env_steps": steps,
            "path": str(destination),
            "sampled_epsilon": epsilon,
            "artifact_id": read_checkpoint_description(destination)["checkpoint_id"],
        }
        self.event(
            "past_copy_exported",
            capture_id=capture_id,
            env_steps=steps,
            path=str(destination),
            sampled_epsilon=epsilon,
            seconds=time.monotonic() - started,
        )

    def validate(self, actor: Path, purpose: str) -> dict[str, Any]:
        """Run or recover one frozen validation task and record its exact result."""
        import jax

        from marl_battlegrounds.training.checkpoints import (
            read_checkpoint_description,
            validation_directory,
        )
        from marl_battlegrounds.training.validation import validate_checkpoint

        assert self.panel is not None
        self.set_status("validation")
        started = time.monotonic()
        validation_options = dict(self.validation_options)
        declared_pairs = None
        if self.validation_declaration is not None:
            from marl_battlegrounds.training.validation import check_confirmation_roots

            root = self.validation_declaration["roots"][purpose]
            if purpose == "confirmation":
                check_confirmation_roots(
                    root,
                    self.host["routine_results"] + self.inherited["records"],
                    used_roots=self.inherited["used_roots"],
                )
            if self.panel.schema_version == 2:
                validation_options["root_seed"] = root
            declared_pairs = self.validation_declaration[
                "routine_seed_pairs"
                if purpose == "routine"
                else "confirmation_seed_pairs"
            ]
        summary = validate_checkpoint(
            actor,
            self.panel,
            output_dir=validation_directory(
                self.root,
                purpose,
                read_checkpoint_description(actor)["metadata"]["checkpoint_id"],
                actor=actor,
            ),
            purpose=purpose,
            seed_pairs=declared_pairs
            if declared_pairs is not None
            else (
                self.config.routine_seed_pairs
                if purpose == "routine"
                else self.config.confirmation_seed_pairs
            ),
            num_envs=32
            if jax.default_backend() == "gpu"
            else min(32, self.config.num_envs),
            chunk_size=128,
            event_callback=lambda record: self.event("validation_segment", **record),
            red_zone_depth=self.config.red_zone_depth,
            **validation_options,
        )
        key = "routine_results" if purpose == "routine" else "confirmation_results"
        if not any(item["task_id"] == summary["task_id"] for item in self.host[key]):
            summary = {
                **summary,
                "elapsed_seconds": self.prior_elapsed + time.monotonic() - self.start,
            }
            self.host[key].append(summary)
            seconds = time.monotonic() - started
            self.host["validation_seconds"] += seconds
            self.host["validation_games"] += summary["games"]
            self.event(
                "validation_complete",
                result=summary,
                seconds=seconds,
            )
        atomic_json(
            self.root / "validation_results.json",
            self.host["routine_results"] + self.host["confirmation_results"],
        )
        self.status["validation_score"] = summary["score"]
        return summary

    def finish_pending(self) -> None:
        """Finish the saved boundary's capture/validation without another decision."""
        actor = None
        if self.config.retain_all_actors:
            known = self.host["actors"].get(self.checkpoint_id)
            actor = self.actor() if known is None else Path(known)
        # Resume may restore the pre-prune list saved at publication. Reconcile
        # it before another save, after repairing any required current export.
        self.prune_recovery()
        pending = self.host["pending"]
        if pending is None:
            return
        if actor is None:
            actor = self.actor()
        if pending.get("routine"):
            self.validate(actor, "routine")
        if pending.get("random"):
            self.validate_random(actor)
        if self.host["env_steps"] == self.config.total_env_steps:
            self.host["final_actor"] = str(actor)
        self.host["pending"] = None
        if pending.get("routine"):
            self.report()

    def validate_random(self, actor: Path) -> None:
        """Complete one fixed Random capture without changing learner or its keys.

        actor is this saved boundary's immutable export. Initialization may reuse
        the fully verified original result configured by the caller. Capture
        timings describe this run just before evaluation/reuse. Original task,
        actor, checkpoint and evaluation pass identities stay unchanged on reuse.
        Publish the accumulated records and existing host game and combat means.
        Combat changes compare equal-map mean kills minus deaths with the
        verified initialization. They are early learning clues, not native wins.
        Games use the run's red_zone_depth; a reused initialization must have
        been recorded at that same depth.
        """
        import jax

        from marl_battlegrounds.training import checkpoints, validation

        pairs = self.config.random_diagnostic_seed_pairs
        assert pairs is not None
        if any(
            row["env_steps"] == self.host["env_steps"]
            for row in self.host["random_results"]
        ):
            return
        self.set_status("random_validation")
        started = time.monotonic()
        elapsed = self.prior_elapsed + started - self.start
        wall = max(0.0, time.time() - self.created_at)
        reference = (
            self.config.random_initialization_result
            if self.host["env_steps"] == 0
            else None
        )
        record: dict[str, Any]
        if reference is None:
            directory = checkpoints.validation_directory(
                self.root,
                "random",
                checkpoints.read_checkpoint_description(actor)["metadata"][
                    "checkpoint_id"
                ],
                actor=actor,
            )
            summary = validation.validate_random(
                actor,
                output_dir=directory,
                seed_pairs=pairs,
                num_envs=32
                if jax.default_backend() == "gpu"
                else min(32, self.config.num_envs),
                chunk_size=128,
                event_callback=lambda record: self.event(
                    "random_validation_segment", **record
                ),
                red_zone_depth=self.config.red_zone_depth,
                **self.validation_options,
            )
            record = {
                **summary,
                "actor_path": str(actor),
                "summary_path": str(directory / "validation_summary.json"),
                "reference_path": None,
                "reused_initialization": False,
            }
            self.host["validation_games"] += summary["games"]
        else:
            identity = checkpoints.artifact_identity(actor)
            record = {
                **validation.read_random_initialization(
                    reference,
                    actor_digest=identity["actor_digest"],
                    seed_pairs=pairs,
                    red_zone_depth=self.config.red_zone_depth,
                    deployment=self.metadata.get("validation_deployment"),
                ),
                "reference_path": str(Path(reference).absolute()),
                "reused_initialization": True,
            }
        record.update(
            elapsed_seconds=elapsed,
            wall_seconds=wall,
            training_seconds=self.host["training_seconds"],
        )
        self.host["random_results"].append(record)
        seconds = time.monotonic() - started
        self.host["validation_seconds"] += seconds
        atomic_json(self.root / "random_diagnostics.json", self.host["random_results"])
        self.event("random_validation_complete", result=record, seconds=seconds)
        self.set_status(
            "random_validation_complete",
            validation_score=record["score"],
            random_validation=_random_progress(self.host["random_results"]),
            **{
                f"validation_{name}": sum(cell[name] for cell in record["cells"])
                for name in ("wins", "draws", "losses")
            },
        )

    def report(self) -> tuple[Path, ...]:
        """Refresh shared reports from durable source records without new evaluation."""
        from marl_battlegrounds.training.analysis import analyze

        self.set_status("reporting")
        started = time.monotonic()
        result = analyze([self.root], output_dir=self.root)
        self.host["report_seconds"] = time.monotonic() - started
        self.event("reports_written", seconds=self.host["report_seconds"])
        return tuple(Path(path) for path in result["artifacts"].values())

    def execute(self) -> TrainResult:
        """Run exact-budget work and preserve failures for explicit resume."""
        import jax
        import numpy as np

        from marl_battlegrounds.training.analysis import (
            confirmation_candidates,
            refresh_summary,
            select_checkpoint,
        )
        from marl_battlegrounds.training.collection import (
            collect_training_rollout,
            training_summary,
        )
        from marl_battlegrounds.training.validation import (
            selection_validation_results,
            validation_root_comparison,
        )

        self.event(
            "attempt_started",
            elapsed_start_seconds=self.prior_elapsed,
            training_start_seconds=self.host["training_seconds"],
            resumed_from=None if self.checkpoint is None else str(self.checkpoint),
            setup_seconds=time.monotonic() - self.start,
            runtime=self.runtime,
            execution=self.metadata["execution"],
        )
        try:
            if self.checkpoint is None:
                self.host["pending"] = {
                    "routine": 0 in self.validation_steps,
                    "random": 0 in self.random_steps,
                }
                self.set_status("initializing")
                self.save()
            self.finish_pending()
            if self.host["env_steps"] < self.config.total_env_steps:
                # Every process prepares its first compiled call, including resume.
                self.set_status("compiling_first_update")
            interval = cast(int, self.config.checkpoint_interval_updates)
            while self.host["env_steps"] < self.config.total_env_steps:
                self._choose_next_games()
                started = time.monotonic()
                collected, rollout = collect_training_rollout(
                    self.collection,
                    self.state.carry,
                    length=self.block_length(),
                    writer=self.writer,
                )
                # The required update-result transfer is the only learner sync.
                next_state, result = self.updater(self.state, collected, rollout)
                host_result = jax.device_get(result)
                seconds = time.monotonic() - started
                accepted = (
                    host_result.performed
                    if is_ppo_method(self.config.method)
                    else host_result.accepted
                )
                if bool(host_result.failed) or not bool(accepted):
                    if self.pqn is not None:
                        self.event(
                            "learner_update_rejected",
                            **_rejected_block_facts(host_result),
                        )
                    raise RuntimeError(
                        "Learner update failed: "
                        f"reason {int(host_result.failure_reason)}"
                    )
                self.state = next_state
                summary, metrics = host_result.summary, host_result.metrics
                updates_before = self.host["completed_updates"]
                self.host["env_steps"] += int(summary.real_transitions)
                self.host["actor_decisions"] += int(summary.live_actor_decisions)
                if self.qmix is not None:
                    self._count_qmix_block(host_result)
                elif self.pqn is not None:
                    self._count_pqn_block(host_result)
                else:
                    self.host["completed_updates"] += 1
                    self._count_ppo_exposure(summary)
                    self.host["used_policy_samples"] += sum(
                        int(x) for x in np.asarray(metrics.actor_samples).flat
                    )
                    self.host["used_value_samples"] += sum(
                        int(x) for x in np.asarray(metrics.critic_samples).flat
                    )
                for name in (
                    "stage_completed",
                    "stage_score_sums",
                    "stage_length_sum",
                    "stage_k20_count",
                ):
                    incoming = np.asarray(getattr(summary, name), dtype=object)
                    previous = np.asarray(
                        self.host.get(name, np.zeros(incoming.shape, dtype=object)),
                        dtype=object,
                    )
                    self.host[name] = (previous + incoming).tolist()
                self._count_member_results(summary)
                self._export_past(host_result.snapshot)
                self.host["training_seconds"] += seconds
                if self.speed_estimate is not None:
                    self.speed_estimate.observe(int(summary.real_transitions), seconds)
                self.host["elapsed_seconds"] = (
                    self.prior_elapsed + time.monotonic() - self.start
                )
                if self.qmix is not None or self.pqn is not None:
                    row = (
                        self._qmix_row(host_result, seconds)
                        if self.qmix is not None
                        else self._pqn_row(host_result, seconds)
                    )
                    self._write_update(row, summary)
                    # Free the block's device arrays before saving or validating.
                    del rollout, collected, result, next_state
                    self._collect_checkpoint_garbage()
                    self._after_block(
                        row,
                        summary,
                        seconds,
                        updates_before,
                        interval,
                        phase="warmup"
                        if self.qmix is not None
                        else "initial_collection",
                    )
                    continue
                row = {
                    "schema_version": 1,
                    "attempt_id": self.metadata["attempt_id"],
                    "update_index": self.host["completed_updates"],
                    "wall_seconds": max(0.0, time.time() - self.created_at),
                    **{
                        key: self.host[key]
                        for key in (
                            "env_steps",
                            "actor_decisions",
                            "used_policy_samples",
                            "used_value_samples",
                            "training_seconds",
                            "elapsed_seconds",
                        )
                    },
                    "collection_update_seconds": seconds,
                    "task_reward_mean": float(summary.task_reward_sum)
                    / int(summary.active_samples)
                    if int(summary.active_samples)
                    else None,
                    "shaping_mean": float(summary.shaping_reward_sum)
                    / int(summary.real_transitions),
                    "policy_loss": float(
                        np.asarray(metrics.actor_loss)[
                            np.asarray(metrics.actor_samples) > 0
                        ].mean()
                    )
                    if np.any(metrics.actor_samples)
                    else None,
                    "value_loss": float(
                        np.asarray(metrics.value_loss)[
                            np.asarray(metrics.critic_samples) > 0
                        ].mean()
                    )
                    if np.any(metrics.critic_samples)
                    else None,
                    "entropy": float(
                        np.asarray(metrics.entropy)[
                            np.asarray(metrics.actor_samples) > 0
                        ].mean()
                    )
                    if np.any(metrics.actor_samples)
                    else None,
                    "learning_rate": self.ppo_reporting.actor_lr,
                    "value_loss_units": "normalized_squared"
                    if self.config.ppo.value_normalization
                    else "reward_squared",
                    **{
                        name: float(
                            np.asarray(getattr(metrics, name))[
                                np.asarray(samples) > 0
                            ].mean()
                        )
                        if np.any(samples)
                        else None
                        for samples, names in (
                            (
                                metrics.actor_samples,
                                (
                                    "approx_kl",
                                    "policy_clip_fraction",
                                    "actor_grad_norm",
                                ),
                            ),
                            (
                                metrics.critic_samples,
                                (
                                    "value_clip_fraction",
                                    "critic_grad_norm",
                                    "target_mean",
                                    "target_std",
                                    "value_mean",
                                    "value_rmse",
                                    "normalization_mean",
                                    "normalization_std",
                                ),
                            ),
                        )
                        for name in names
                    },
                }
                self._write_update(row, summary)
                del rollout, collected, result, next_state
                self._collect_checkpoint_garbage()
                steps = self.host["env_steps"]
                capture = (
                    steps in self.validation_steps
                    or steps in self.config.checkpoint_env_steps
                    or steps == self.config.total_env_steps
                )
                if capture:
                    self.host["pending"] = {
                        "routine": steps in self.validation_steps,
                        "random": steps in self.random_steps,
                    }
                if capture or _crosses_interval(
                    updates_before, self.host["completed_updates"], interval
                ):
                    self.save()
                    self.finish_pending()
                self.set_status(
                    "training",
                    recent_transitions_per_second=int(summary.real_transitions)
                    / seconds,
                    **{
                        key: row[key]
                        for key in (
                            "policy_loss",
                            "value_loss",
                            "entropy",
                            "task_reward_mean",
                            "shaping_mean",
                        )
                    },
                )
            if self.panel is not None:
                final_id = next(
                    key
                    for key, path in self.host["actors"].items()
                    if path == self.host["final_actor"]
                )
                actor_paths = {
                    **{
                        key: value["actor_path"]
                        for key, value in self.inherited["actors"].items()
                    },
                    **self.host["actors"],
                }
                rule = self.metadata.get("selection_rule", "saved")
                routine = self.host["routine_results"]
                if self.validation_declaration is None:
                    candidate_ids = confirmation_candidates(
                        self.host["routine_results"],
                        final_checkpoint_id=final_id,
                        rule=rule,
                    )
                    confirmations = self.host["confirmation_results"]
                else:
                    _, confirmations, candidate_ids = selection_validation_results(
                        self.host["routine_results"],
                        self.host["confirmation_results"],
                        self.inherited,
                        declaration=self.validation_declaration,
                        panel=self.panel,
                        final_checkpoint_id=final_id,
                        rule=rule,
                    )
                completed = {row["checkpoint_id"] for row in confirmations}
                for identifier in candidate_ids:
                    if identifier not in completed:
                        self.validate(Path(actor_paths[identifier]), "confirmation")
                if self.validation_declaration is not None:
                    routine, confirmations, _ = selection_validation_results(
                        self.host["routine_results"],
                        self.host["confirmation_results"],
                        self.inherited,
                        declaration=self.validation_declaration,
                        panel=self.panel,
                        final_checkpoint_id=final_id,
                        rule=rule,
                    )
                if candidate_ids:
                    selected = select_checkpoint(confirmations, rule=rule)
                    if self.validation_declaration is not None:
                        selected.update(
                            routine_comparison=validation_root_comparison(
                                routine,
                                allow_different_roots=self.validation_declaration[
                                    "allow_different_roots"
                                ],
                            ),
                            confirmation_comparison=validation_root_comparison(
                                confirmations,
                                allow_different_roots=self.validation_declaration[
                                    "allow_different_roots"
                                ],
                            ),
                        )
                    self.host["selection"] = selected
                    self.host["selected_actor"] = actor_paths[selected["checkpoint_id"]]
                    atomic_json(self.root / "selection.json", selected)
                else:
                    # A declared child may end before its first learning update.
                    # Its final actor still loads, but it is not selected evidence.
                    reason = "No eligible trained checkpoint"
                    self.status["selection_unavailable_reason"] = reason
                    self.event("selection_unavailable", reason=reason)
            if self.config.slot_diagnostic:
                self.run_slot_check()
            exposure = training_summary(self.collection, self.state.carry)
            if is_ppo_method(self.config.method):
                exposure["ppo_exposure"] = self.host["ppo_exposure"]
            if self.qmix is not None:
                exposure["sampled_exposure"] = {
                    "unit": "TD pairs used by optimizer steps; repeats count again",
                    "used_td_pairs": self.host["used_td_pairs"],
                    **self.host["sampled_exposure"],
                }
            if self.pqn is not None:
                exposure["used_exposure"] = {
                    "unit": (
                        "TD pairs used by optimizer steps, by left row; each "
                        "epoch counts again and kept rows are used again"
                    ),
                    "used_td_pairs": self.host["used_td_pairs"],
                    "used_prefix_td_pairs": self.host["used_prefix_td_pairs"],
                    **self.host["used_exposure"],
                }
            exposure.update(
                {
                    name: self.host.get(name)
                    for name in (
                        "stage_completed",
                        "stage_score_sums",
                        "stage_length_sum",
                        "stage_k20_count",
                    )
                }
            )
            thresholds = cast(list[int], exposure["score_thresholds_by_episode_stage"])
            completed = self.host.get("stage_completed")
            if completed is not None:
                for row in cast(
                    list[dict[str, Any]], exposure["exposure_by_score_threshold"]
                ):
                    counts = [
                        sum(
                            completed[i][outcome]
                            for i, threshold in enumerate(thresholds)
                            if threshold == row["score_threshold"]
                        )
                        for outcome in range(3)
                    ]
                    row.update(
                        completed_games=sum(counts),
                        wins=counts[0],
                        draws=counts[1],
                        losses=counts[2],
                    )
            atomic_json(self.root / "exposure.json", exposure)
            self.set_status("finalizing")
            paths = self.report()
            if not all(path.is_file() and path.stat().st_size for path in paths):
                raise ValueError("Required final reports are missing or empty")
            memory = _memory_snapshot()
            self.set_status(
                "complete",
                status="complete",
                pending_work_seconds=0.0,
                exit_code=0,
                memory=memory,
                evidence_paths=[
                    path.relative_to(self.root).as_posix() for path in paths
                ],
            )
            refresh_summary([self.root], output_dir=self.root)
            self.event("attempt_complete", memory=memory)
            return TrainResult(
                self.root,
                Path(self.host["final_actor"]),
                None
                if self.host["selected_actor"] is None
                else Path(self.host["selected_actor"]),
                self.host["env_steps"],
                self.host["completed_updates"],
                "complete",
                paths,
                cast(Path, self.checkpoint),
            )
        except BaseException as error:
            memory = _memory_snapshot()
            self.event(
                "attempt_failed",
                error=f"{type(error).__name__}: {error}",
                memory=memory,
            )
            self.set_status(
                "failed",
                status="failed",
                error=str(error),
                exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
                memory=memory,
            )
            try:
                refresh_summary([self.root], output_dir=self.root)
            except Exception as report_error:
                self.event("failure_summary_unavailable", error=str(report_error))
            raise

    def _count_qmix_block(self, result: Any) -> None:  # noqa: ANN401
        """Add one accepted QMIX block to the exact host totals.

        result is the host copy of a QMIXUpdateResult. A warmup block adds a
        block only; a learning block also adds epochs optimizer steps and its
        sampled counts. Totals are Python integers and may exceed int32.
        """
        import numpy as np

        assert self.qmix is not None
        self.host["completed_blocks"] += 1
        if bool(result.performed):
            self.host["learning_blocks"] += 1
            self.host["completed_updates"] += self.qmix.epochs
        sampled = result.sampled
        for name in ("sampled_sequences", "used_td_pairs", "used_agent_utilities"):
            self.host[name] += int(getattr(sampled, name))
        exposure = self.host["sampled_exposure"]
        for key, field_name in (
            ("by_stage", "exposure_by_stage"),
            ("by_source", "exposure_by_source"),
            ("by_opponent", "exposure_by_opponent"),
        ):
            counts = cast(list[int], np.asarray(getattr(sampled, field_name)).tolist())
            totals = cast(list[int], exposure[key])
            exposure[key] = [
                total + int(count) for total, count in zip(totals, counts, strict=True)
            ]

    def _qmix_row(self, result: Any, seconds: float) -> dict[str, Any]:  # noqa: ANN401
        """Build one QMIX training-log row from host totals and a block result.

        result is the host copy of the block's QMIXUpdateResult, already added
        to the host totals by _count_qmix_block; seconds is the block's
        collection-plus-update wall time in seconds. Returns the JSON-ready row
        for training_updates.jsonl: the shared counters, the exact QMIX totals,
        task and shaping reward means, replay_rows_per_lane (rows stored per
        game, at most buffer_size), td_pairs_per_transition (used TD pairs per
        real transition; repeats count again), loss, mean_q and mean_target
        (averages over the block's optimizer steps; None for a warmup block),
        grad_norm_mean, grad_norm_max and grad_norm_nonfinite (the raw,
        unclipped global gradient norms of the steps that ran, summarized by
        _gradient_norm_summary: None, None and 0 for a warmup block), epsilon
        (the clock's rate after the block, a float64 host value) and
        learning_rate. No PPO loss or sample keys appear. Reads host values
        only; no device work.
        """
        import numpy as np

        from marl_battlegrounds.baselines.qmix import epsilon_reference

        assert self.qmix is not None
        summary, metrics = result.summary, result.metrics
        learned = bool(result.performed)
        rounds = self.host["env_steps"] // self.config.num_envs

        def mean(values: object) -> float | None:
            """Average one per-epoch metric over this block's optimizer steps."""
            return float(np.mean(np.asarray(values))) if learned else None

        return {
            "schema_version": 1,
            "attempt_id": self.metadata["attempt_id"],
            "wall_seconds": max(0.0, time.time() - self.created_at),
            **{
                key: self.host[key]
                for key in (
                    "env_steps",
                    "actor_decisions",
                    "completed_blocks",
                    "learning_blocks",
                    "completed_updates",
                    "sampled_sequences",
                    "used_td_pairs",
                    "used_agent_utilities",
                    "training_seconds",
                    "elapsed_seconds",
                )
            },
            "collection_update_seconds": seconds,
            "task_reward_mean": float(summary.task_reward_sum)
            / int(summary.active_samples)
            if int(summary.active_samples)
            else None,
            "shaping_mean": float(summary.shaping_reward_sum)
            / int(summary.real_transitions),
            "replay_rows_per_lane": min(rounds, self.qmix.buffer_size),
            "td_pairs_per_transition": self.host["used_td_pairs"]
            / self.host["env_steps"],
            "loss": mean(metrics.loss),
            "mean_q": mean(metrics.mean_q),
            "mean_target": mean(metrics.mean_target),
            # A step with no eligible TD pair applies no update; leave it out.
            **_gradient_norm_summary(
                metrics.grad_norm,
                learned & (np.asarray(metrics.used_td_pairs) > 0),
            ),
            "epsilon": self.continuation.exploration.at(self.host["env_steps"])
            if self.continuation is not None
            and self.continuation.exploration is not None
            else epsilon_reference(
                rounds, self.config.num_envs, self.qmix.eps_min, self.qmix.eps_decay
            ),
            "learning_rate": cast(QMIXConfig, self.qmix_reporting).q_lr,
        }

    def _count_pqn_block(self, result: Any) -> None:  # noqa: ANN401
        """Add one accepted PQN-VDN block to the exact host totals.

        result is the host copy of a PQNUpdateResult. Every accepted block
        adds one completed block; a learning block (performed) also adds a
        learning block, epochs * num_minibatches optimizer steps and its used
        counts and exposure. Totals are Python integers and may exceed int32.
        """
        import numpy as np

        assert self.pqn is not None
        self.host["completed_blocks"] += 1
        if bool(result.performed):
            self.host["learning_blocks"] += 1
            self.host["completed_updates"] += self.pqn.epochs * self.pqn.num_minibatches
        used = result.used
        for name in (
            "used_sequences",
            "used_td_pairs",
            "used_agent_utilities",
            "used_prefix_td_pairs",
        ):
            self.host[name] += int(getattr(used, name))
        exposure = self.host["used_exposure"]
        for key, field_name in (
            ("by_stage", "exposure_by_stage"),
            ("by_source", "exposure_by_source"),
            ("by_opponent", "exposure_by_opponent"),
        ):
            counts = cast(list[int], np.asarray(getattr(used, field_name)).tolist())
            totals = cast(list[int], exposure[key])
            exposure[key] = [
                total + int(count) for total, count in zip(totals, counts, strict=True)
            ]

    def _pqn_row(self, result: Any, seconds: float) -> dict[str, Any]:  # noqa: ANN401
        """Build one PQN-VDN training-log row from host totals and a block result.

        Parameters
        ----------
        result : PQNUpdateResult
            Host copy of the block's result, already added to the host totals
            by _count_pqn_block.
        seconds : float
            The block's collection-plus-update wall time in seconds.

        Returns
        -------
        dict
            JSON-ready row for training_updates.jsonl: the shared counters,
            completed_blocks, learning_blocks, completed_updates (optimizer
            steps), initial_rounds_completed (at most W), the exact used
            totals and used_exposure (cumulative, like the other counts),
            task and shaping reward means, and loss, mean_q and mean_target
            (averages over the block's minibatch steps; None for an initial
            chunk). grad_norm_mean, grad_norm_max and grad_norm_nonfinite
            summarize the global gradient norms measured before clipping on
            the steps that ran (_gradient_norm_summary; None, None and 0 for
            an initial chunk). epsilon is the rate current actors hold at this boundary
            (1.0 before W, then the block clock's value, a float64 host
            value); learning_rate is the scheduled rate of the block's last
            applied step (None for an initial chunk). No PPO or QMIX-only keys.

        Notes
        -----
        Reads host values only; adds no device work or synchronization.
        """
        import numpy as np

        from marl_battlegrounds.baselines.pqn import (
            pqn_epsilon_reference,
            pqn_learning_rate,
        )

        assert self.pqn is not None
        summary, metrics = result.summary, result.metrics
        learned = bool(result.performed)
        rounds = self.host["env_steps"] // self.config.num_envs
        planned = self.planned_learning_blocks

        def mean(values: object) -> float | None:
            """Average one (epochs, minibatches) metric over this block's steps."""
            return float(np.mean(np.asarray(values))) if learned else None

        return {
            "schema_version": 1,
            "attempt_id": self.metadata["attempt_id"],
            "wall_seconds": max(0.0, time.time() - self.created_at),
            **{
                key: self.host[key]
                for key in (
                    "env_steps",
                    "actor_decisions",
                    "completed_blocks",
                    "learning_blocks",
                    "completed_updates",
                )
            },
            "initial_rounds_completed": min(rounds, self.pqn.initial_rounds),
            **{
                key: self.host[key]
                for key in (
                    "used_sequences",
                    "used_td_pairs",
                    "used_agent_utilities",
                    "used_prefix_td_pairs",
                    "used_exposure",
                    "training_seconds",
                    "elapsed_seconds",
                )
            },
            "collection_update_seconds": seconds,
            "task_reward_mean": float(summary.task_reward_sum)
            / int(summary.active_samples)
            if int(summary.active_samples)
            else None,
            "shaping_mean": float(summary.shaping_reward_sum)
            / int(summary.real_transitions),
            "loss": mean(metrics.loss),
            "mean_q": mean(metrics.mean_q),
            "mean_target": mean(metrics.mean_target),
            **_gradient_norm_summary(
                metrics.grad_norm, learned & np.asarray(metrics.performed)
            ),
            "epsilon": 1.0
            if rounds < self.pqn.initial_rounds
            else self.continuation.exploration.at(self.host["learning_blocks"])
            if self.continuation is not None
            and self.continuation.exploration is not None
            else pqn_epsilon_reference(self.host["learning_blocks"], planned, self.pqn),
            "learning_rate": (
                self.continuation.pqn_rate.at(self.host["completed_updates"] - 1)
                if self.continuation is not None
                and self.continuation.pqn_rate is not None
                else pqn_learning_rate(
                    self.host["completed_updates"] - 1, planned, self.pqn
                )
            )
            if learned
            else None,
        }

    def block_length(self) -> int:
        """Return the rounds to request for the next collection block.

        PPO and QMIX always request rollout_length. PQN-VDN requests
        ``min(T, W - r)`` during its W initial rounds (a chunk of T, then one
        of H) and T afterwards; collection stops at the exact budget, so a
        final block may hold fewer real rounds. Reads host counts only.
        """
        if self.pqn is None:
            return self.rollout_length
        from marl_battlegrounds.training._continuation_schedules import (
            reward_refill_end,
        )

        rounds = self.host["env_steps"] // self.config.num_envs
        refill_end = reward_refill_end(
            self.continuation,
            initial_rounds=self.pqn.initial_rounds,
            memory_window=self.pqn.memory_window,
        )
        return (
            min(self.rollout_length, refill_end - rounds)
            if rounds < refill_end
            else self.rollout_length
        )

    def _count_ppo_exposure(self, summary: Any) -> None:  # noqa: ANN401
        """Add optimizer-used samples from the same transferred update result.

        Epochs already enter the device totals; minibatches are disjoint parts.
        Python sums do not overflow. A resumed legacy prefix remains labelled
        unknown through the saved offsets rather than filled with guessed counts.
        """
        for kind in ("policy", "value"):
            for dimension in ("source", "opponent"):
                key = f"{kind}_{dimension}"
                incoming = getattr(summary, f"used_{kind}_by_{dimension}").tolist()
                previous = self.host["ppo_exposure"][key]
                self.host["ppo_exposure"][key] = [
                    int(old) + int(new)
                    for old, new in zip(previous, incoming, strict=True)
                ]

    def _count_member_results(self, summary: Any) -> None:  # noqa: ANN401
        """Add one block's completed-game counts using unbounded Python integers."""
        for key, summary_field in (
            ("member_completed", "per_member_completed"),
            ("member_score_sums", "per_member_score_sums"),
        ):
            incoming = getattr(summary, summary_field).tolist()
            self.host["last_" + key] = incoming
            previous = self.host[key]
            if len(previous) != len(incoming):
                raise ValueError("Member result rows differ from the saved population")
            self.host[key] = [
                [int(old) + int(new) for old, new in zip(left, right, strict=True)]
                for left, right in zip(previous, incoming, strict=True)
            ]
        incoming = summary.per_member_length_sum.tolist()
        self.host["last_member_length_sum"] = incoming
        previous = self.host["member_length_sum"]
        if previous is not None:
            self.host["member_length_sum"] = [
                int(old) + int(new) for old, new in zip(previous, incoming, strict=True)
            ]

    def _member_statistics(
        self, *, completed_only: bool = False
    ) -> list[dict[str, Any]]:
        """Attach readable identities to existing cumulative and last-block counts.

        Zero-game margins are None, never invented measurements. Logging asks
        for completed_only to avoid serializing every old capture every update;
        an enabled researcher rule can inspect the full declared population.
        Cumulative counts start at start_env_steps; complete_history=False means
        an old checkpoint did not record its earlier member outcomes. These
        observed suffix counts remain useful to callbacks, but cannot describe
        full-run exposure. Unknown legacy lengths remain None.
        """
        from marl_battlegrounds.training.collection import opponent_member_records

        def counts(
            outcomes: list[int], points: list[int], length: int | None
        ) -> dict[str, Any]:
            """Describe Team A results; every native point retains its team owner."""
            games = sum(outcomes)
            return dict(
                games=games,
                completed_games=games,
                length_sum=length,
                mean_length=length / games if games and length is not None else None,
                length_unavailable_reason=(
                    "Legacy checkpoint did not record member lengths"
                    if length is None
                    else None
                ),
                wins=outcomes[0],
                draws=outcomes[1],
                losses=outcomes[2],
                points_for=points[0],
                points_against=points[1],
                mean_point_margin=(points[0] - points[1]) / games if games else None,
            )

        records = opponent_member_records(self.collection, self.state.carry)
        result: list[dict[str, Any]] = []
        for index, member in enumerate(records):
            last = self.host["last_member_completed"][index]
            if completed_only and not sum(last):
                continue
            capture = self.host["past_copies"].get(str(member.get("capture_id")), {})
            result.append(
                {
                    **member,
                    **capture,
                    "cumulative": {
                        **counts(
                            self.host["member_completed"][index],
                            self.host["member_score_sums"][index],
                            None
                            if self.host["member_length_sum"] is None
                            else self.host["member_length_sum"][index],
                        ),
                        "start_env_steps": self.host.get(
                            "member_results_start_env_steps", 0
                        ),
                        "complete_history": self.host.get(
                            "member_results_start_env_steps", 0
                        )
                        == 0,
                    },
                    "last_update": {
                        **counts(
                            last,
                            self.host["last_member_score_sums"][index],
                            None
                            if self.host["last_member_length_sum"] is None
                            else self.host["last_member_length_sum"][index],
                        ),
                        "recorded": self.host.get("env_steps", 0)
                        > self.host.get("member_results_start_env_steps", 0),
                    },
                }
            )
        return result

    def _choose_next_games(self) -> None:
        """Apply a researcher rule between updates without changing running games."""
        if self.matchmaking is None:
            return
        from marl_battlegrounds.training.collection import (
            update_opponent_selection,
            update_partner_selection,
        )

        choice = self.matchmaking(
            deepcopy(
                {
                    "members": self._member_statistics(),
                    "partners": list(self.collection.partner_records),
                }
            ),
            self.host["env_steps"],
        )
        if not isinstance(cast(object, choice), Mapping) or set(choice) - {
            "opponents",
            "past",
            "partners",
        }:
            raise ValueError(
                "matchmaking must return opponents, partners and/or past weights"
            )
        carry = self.state.carry
        if "opponents" in choice or "past" in choice:
            self.collection, carry = update_opponent_selection(
                self.collection,
                carry,
                selection=cast(
                    Mapping[str, float] | Sequence[str] | None, choice.get("opponents")
                ),
                past=cast(Mapping[int, float] | None, choice.get("past")),
            )
        self.collection, carry = update_partner_selection(
            self.collection,
            carry,
            selection=cast(
                Mapping[str, float] | Sequence[str] | None, choice.get("partners")
            ),
        )
        self.state = self.state._replace(carry=carry)

    def _write_update(self, row: dict[str, Any], summary: Any) -> None:  # noqa: ANN401
        """Save one finite update row, then call the optional researcher logger.

        Add the stable run ID to the existing step and attempt fields. Enabled
        custom feedback has its own mean over owned active samples.
        The callback receives a deep copy after the normal file append; changing
        it cannot change bookkeeping. Disabled callbacks copy nothing. Callback
        errors propagate with the saved row intact; normal recovery may replay
        work, so external services must deduplicate rather than assume one call.
        Named custom components transfer only a small collection window here.
        Prepare its finite row/totals first; only a successful append folds host
        totals and clears the device window, before calling the logger. Earlier
        checkpoints keep pending windows. Recovery restores those totals and
        rewinds the abandoned row suffix. No per-step callback is introduced.
        """
        if summary.custom_reward_sum is not None:
            row["custom_reward_mean"] = (
                float(summary.custom_reward_sum) / int(summary.active_samples)
                if int(summary.active_samples)
                else None
            )
        row["run_id"] = self.metadata["run_id"]
        row["opponent_results"] = self._member_statistics(completed_only=True)
        component_record = None
        cleared_carry = None
        if self.collection.reward_component_keys:
            import jax
            import jax.numpy as jnp

            from marl_battlegrounds.training.shaping import (
                _merge_reward_records,  # pyright: ignore[reportPrivateUsage]
                _reward_totals_record,  # pyright: ignore[reportPrivateUsage]
                _RewardTotals,  # pyright: ignore[reportPrivateUsage]
            )

            totals = self.state.carry.reward_totals
            assert isinstance(totals, _RewardTotals)
            delta = _reward_totals_record(totals, self.collection.reward_component_keys)
            row["collected_reward_components"] = delta
            component_record = _merge_reward_records(
                self.host.get("collected_reward_components"), delta
            )

            def zero(value: Array) -> Array:
                """Clear only the small accepted accounting window on its device."""
                return jnp.zeros_like(value)

            cleared_carry = self.state.carry._replace(
                reward_totals=jax.tree.map(zero, totals)
            )
        append_jsonl(self.root / "training_updates.jsonl", row)
        if component_record is not None:
            self.host["collected_reward_components"] = component_record
            self.state = self.state._replace(carry=cleared_carry)
        if self.on_update is not None:
            try:
                self.on_update(deepcopy(row))
            except Exception as error:
                raise RuntimeError(
                    "on_update failed after saving training_updates.jsonl; "
                    "resume from the latest complete checkpoint"
                ) from error

    def _after_block(
        self,
        row: dict[str, Any],
        summary: Any,  # noqa: ANN401
        seconds: float,
        updates_before: int,
        interval: int,
        *,
        phase: str,
    ) -> None:
        """Apply the shared capture and save rules after one QMIX or PQN-VDN block.

        row is the block's log row from _qmix_row or _pqn_row; summary is its
        host UpdateSummary; seconds is the block's collection-plus-update
        time; updates_before is the optimizer-step total before the block;
        interval is the resolved checkpoint_interval_updates; phase is the
        status name before the first learning block ("warmup" for QMIX,
        "initial_collection" for PQN-VDN). A capture point (a validation
        fraction, an extra save point or the final step) or a crossed
        multiple of the interval saves the learner and finishes pending
        captures, exactly as for PPO; blocks without optimizer steps never
        cross an interval. Status reports phase until the first learning
        block, then "training", with the row's loss, Q values, reward means,
        epsilon and the learning-block count. Writes checkpoint and status
        files.
        """
        steps = self.host["env_steps"]
        capture = (
            steps in self.validation_steps
            or steps in self.config.checkpoint_env_steps
            or steps == self.config.total_env_steps
        )
        if capture:
            self.host["pending"] = {
                "routine": steps in self.validation_steps,
                "random": steps in self.random_steps,
            }
        if capture or _crosses_interval(
            updates_before, self.host["completed_updates"], interval
        ):
            self.save()
            self.finish_pending()
        self.set_status(
            "training" if self.host["learning_blocks"] else phase,
            recent_transitions_per_second=int(summary.real_transitions) / seconds,
            learning_blocks=self.host["learning_blocks"],
            **{
                key: row[key]
                for key in (
                    "loss",
                    "mean_q",
                    "task_reward_mean",
                    "shaping_mean",
                    "epsilon",
                )
            },
        )

    def run_slot_check(self) -> None:
        """Run the declared trained comparison only after the final actor exists.

        The games use the run's red_zone_depth, which the slot task records.
        """
        from marl_battlegrounds.training.validation import run_slot_diagnostic

        assert self.panel is not None
        self.set_status("slot_diagnostic")
        started = time.monotonic()
        opponent = (
            None
            if self.config.slot_diagnostic_actor is None
            else _local_actor(self.config.slot_diagnostic_actor)
        ) or self.panel.members[-1].path
        assert opponent is not None
        result = run_slot_diagnostic(
            Path(self.host["final_actor"]),
            opponent,
            output_dir=self.root
            / "slot_diagnostic"
            / Path(self.host["final_actor"]).name,
            event_callback=lambda record: self.event("slot_segment", **record),
            red_zone_depth=self.config.red_zone_depth,
        )
        atomic_json(self.root / "slot_diagnostic.json", result)
        self.host["slot_complete"] = True
        self.event("slot_diagnostic_finished", seconds=time.monotonic() - started)
