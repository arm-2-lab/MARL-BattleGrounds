"""Save and load portable model files: one agent's decision in one JAX file.

A portable model file holds everything needed to play one trained actor
without its training code: the agent's decision program (exported with
``jax.export``), its weights, its starting memory and the original actor
description. ``load_portable_model`` turns a file into an ordinary JAX
``System`` that plays a whole team by running the program once per agent, so
it works everywhere a System works: ``evaluate``, ``pool``, ``team``,
tournaments, training opponents and the BattleClient. It runs under ``jit``
and under your own ``jax.vmap``. ``export_model`` writes a file from a saved
built-in actor folder or from any per-agent JAX ``Policy``.

Who owns what. This module owns the file format (``marl-bgs-portable-model``,
format version 1) and the loader System. ``released_models`` owns released
names and downloads. The per-agent decision functions for the built-in
families (recurrent and feedforward PPO, QMIX, PQN-VDN) repeat the per-agent
part of the frozen team hooks in ``baselines``. That repetition is deliberate:
the team hooks are frozen, and the per-call tests compare both on the same
inputs.

File layout: an uncompressed ZIP with exactly these parts.

==========================  ===================================================
Part                        Holds
==========================  ===================================================
``model.json``              Format, method, step, identities, every array's
                            path, shape and dtype, and each other part's SHA-256
``decide.jaxexport``        The agent's decision program, for cpu, cuda, rocm
                            and tpu, with an open number of rows
``weights.npz``             The weights, keyed by their path
``initial_memory.npz``      One agent's starting memory, keyed by its path
``actor_details.json``      The original actor description, byte for byte
==========================  ===================================================

The decision program takes ``(weights, memory, actor_input, action_mask,
key_data)`` for R independent agent rows and returns the chosen action (three
int32 heads), the next memory and the teacher outputs: the 198 joint-action
log-probabilities (PPO) or action values (QMIX, PQN-VDN) in the world frame,
with illegal actions at negative infinity. In programs written by
``export_model``, rows never see each other, so each agent's information limits
hold by construction. The loader cannot prove this for a program made some
other way: such a program gets all R rows in one call.

Loading needs only JAX, NumPy and the ``flatbuffers`` package; exporting a
built-in actor folder also needs the training extra.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    _declare_lane_independent,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.evaluation.recording_identity import tree_digest
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput

if TYPE_CHECKING:
    from jax import export as jax_export

PORTABLE_MODEL_FORMAT = "marl-bgs-portable-model"
PORTABLE_MODEL_FORMAT_VERSION = 1
PLATFORMS = ("cpu", "cuda", "rocm", "tpu")

_PARTS = (
    "model.json",
    "decide.jaxexport",
    "weights.npz",
    "initial_memory.npz",
    "actor_details.json",
)
_NUM_ACTIONS = 198
_TEAM_SLOTS = 5
# Fixed ZIP timestamps make a file's bytes depend only on its content.
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_TEACHER_KINDS = {
    "mappo": "log_probabilities",
    "ippo": "log_probabilities",
    "ff_mappo": "log_probabilities",
    "ff_ippo": "log_probabilities",
    "qmix": "action_values",
    "pqn_vdn": "action_values",
}


@dataclass(frozen=True)
class PortableModel:
    """The checked contents of one portable model file, without its program.

    Attributes
    ----------
    path : Path
        The file that was read, as given by the caller (not resolved).
    sha256 : str
        Lowercase hex SHA-256 of the whole file.
    model : dict[str, Any]
        Parsed ``model.json``. Keys include ``format``, ``format_version``,
        ``game_mode``, ``method``, ``display_name``, ``training_step`` (int, or
        None for a file made from a Policy), ``checkpoint_id``,
        ``inference_digest``, ``actor_digest``, ``inputs``, ``memory``,
        ``weights``, ``outputs``, ``platforms`` and ``parts`` (part name to
        SHA-256).
    actor_details : dict[str, Any]
        Parsed ``actor_details.json``: the original actor description for a
        built-in actor, or a short description for a file made from a Policy.

    Notes
    -----
    A plain record. Building it does not load the program or the weights.
    """

    path: Path
    sha256: str
    model: dict[str, Any]
    actor_details: dict[str, Any]


class PortableVariables(NamedTuple):
    """Numerical inputs of a loaded portable System, kept dynamic under jit.

    Attributes
    ----------
    weights : tuple[Array, ...]
        The weights in ``model.json`` order, unbatched.
    initial_memory : tuple[Array, ...]
        One agent's starting memory leaves in ``model.json`` order, unbatched.
        Empty for a model without memory.

    Notes
    -----
    Replacing these with same-shaped arrays (for example another save of the
    same model) reuses the compiled program.
    """

    weights: tuple[Array, ...]
    initial_memory: tuple[Array, ...]


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------


def _read_parts(path: Path) -> tuple[str, dict[str, bytes]]:
    """Read every part of a portable model ZIP after checking its member list.

    Returns the whole file's SHA-256 and the part bytes by name. Raises
    ValueError when the file is not a ZIP, when its members are not exactly the
    five parts, or when a member is compressed or nested. Reads only the
    listed members; no path inside the ZIP is ever used as a file name.
    """
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise ValueError(f"{path} is not a portable model file (not a ZIP)") from error
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if sorted(names) != sorted(_PARTS) or len(set(names)) != len(names):
            raise ValueError(
                f"{path} is not a portable model file: expected the parts "
                f"{', '.join(_PARTS)}, found {', '.join(names) or 'nothing'}"
            )
        if any(info.compress_type != zipfile.ZIP_STORED for info in infos):
            raise ValueError(f"{path} has compressed parts; format 1 stores them")
        return digest, {name: archive.read(name) for name in _PARTS}


def _check_model(path: Path, model: object, parts: Mapping[str, bytes]) -> None:
    """Check model.json's format fields and every other part's SHA-256."""
    if not isinstance(model, dict):
        raise ValueError(f"{path}: model.json is not a JSON object")
    record = cast(dict[str, Any], model)
    if record.get("format") != PORTABLE_MODEL_FORMAT:
        raise ValueError(f"{path}: model.json does not name {PORTABLE_MODEL_FORMAT}")
    if record.get("format_version") != PORTABLE_MODEL_FORMAT_VERSION:
        raise ValueError(
            f"{path}: format version {record.get('format_version')!r} is not "
            f"supported; this MARL-BGs reads version {PORTABLE_MODEL_FORMAT_VERSION}"
        )
    hashes = cast(object, record.get("parts"))
    if not isinstance(hashes, dict) or set(cast(dict[str, str], hashes)) != set(
        _PARTS[1:]
    ):
        raise ValueError(f"{path}: model.json must list the SHA-256 of every part")
    for name in _PARTS[1:]:
        if (
            hashlib.sha256(parts[name]).hexdigest()
            != cast(dict[str, str], hashes)[name]
        ):
            raise ValueError(f"{path}: part {name} does not match its SHA-256")


def is_portable_model_file(path: str | Path) -> bool:
    """Say whether a path is a portable model file, without checking its parts.

    Parameters
    ----------
    path : str or Path
        Any path. Folders, missing paths and other files return False.

    Returns
    -------
    bool
        True when the path is a ZIP file whose ``model.json`` names the
        ``marl-bgs-portable-model`` format. Part hashes are not checked here;
        ``read_portable_model`` and ``load_portable_model`` check them.

    Notes
    -----
    Host-only and cheap: reads the ZIP directory and one small member. A
    compressed ``model.json`` or one larger than 1 MiB returns False without
    being read, so a small crafted file cannot use up memory here.
    """
    candidate = Path(path)
    if not candidate.is_file() or not zipfile.is_zipfile(candidate):
        return False
    try:
        with zipfile.ZipFile(candidate) as archive:
            if "model.json" not in archive.namelist():
                return False
            info = archive.getinfo("model.json")
            if info.compress_type != zipfile.ZIP_STORED or info.file_size > 1 << 20:
                return False
            model = cast(object, json.loads(archive.read("model.json")))
    except (OSError, ValueError, zipfile.BadZipFile):
        return False
    return (
        isinstance(model, dict)
        and cast(dict[str, object], model).get("format") == PORTABLE_MODEL_FORMAT
    )


def read_portable_model(path: str | Path) -> PortableModel:
    """Read and check a portable model file's description and identities.

    Parameters
    ----------
    path : str or Path
        A portable model file (``.zip``).

    Returns
    -------
    PortableModel
        The file's SHA-256, its parsed ``model.json`` and its parsed
        ``actor_details.json``. The program and weights are checked but not
        loaded.

    Raises
    ------
    ValueError
        The file is not a format-1 portable model file, a part is missing,
        extra or compressed, or a part does not match its SHA-256.
    OSError
        The file cannot be read.

    Notes
    -----
    Host-only. Reads the whole file once; writes nothing.
    """
    source = Path(path)
    digest, parts = _read_parts(source)
    model = json.loads(parts["model.json"])
    _check_model(source, model, parts)
    details = json.loads(parts["actor_details.json"])
    if not isinstance(details, dict):
        raise ValueError(f"{source}: actor_details.json is not a JSON object")
    return PortableModel(
        source, digest, cast(dict[str, Any], model), cast(dict[str, Any], details)
    )


def _arrays(
    blob: bytes, rows: Sequence[Mapping[str, Any]], what: str
) -> tuple[Array, ...]:
    """Read named NumPy arrays in model.json order and check shape and dtype.

    blob is an uncompressed ``.npz`` part. A compressed member is refused
    before any array is read (ValueError), so reading never needs more memory
    than the part itself.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            stored = all(
                info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
            )
    except zipfile.BadZipFile as error:
        raise ValueError(f"{what} are not an .npz file") from error
    if not stored:
        raise ValueError(f"{what} have compressed arrays; format 1 stores them")
    with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
        if sorted(saved.files) != sorted(str(row["path"]) for row in rows):
            raise ValueError(f"{what} do not match the arrays model.json lists")
        values: list[Array] = []
        for row in rows:
            value = saved[str(row["path"])]
            if (
                list(value.shape) != list(row["shape"])
                or str(value.dtype) != row["dtype"]
            ):
                raise ValueError(f"{what}: {row['path']} has the wrong shape or dtype")
            values.append(jnp.asarray(value))
    return tuple(values)


def _nested(paths: Sequence[str], leaves: Sequence[Array]) -> dict[str, Any]:
    """Build a nested dictionary from slash-separated key paths and leaves."""
    tree: dict[str, Any] = {}
    for text, leaf in zip(paths, leaves, strict=True):
        node = tree
        *parents, last = text.split("/")
        for key in parents:
            node = node.setdefault(key, {})
        node[last] = leaf
    return tree


def portable_actor_variables(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Rebuild a built-in actor's weights from a portable file, to train from them.

    Parameters
    ----------
    path : str or Path
        A portable model file made from a saved built-in actor.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        The actor's Flax variable tree, exactly as the actor folder stores it
        (nested dictionaries, for example ``{"params": {...}}``; PQN-VDN adds
        ``"batch_stats"``), and the original ``actor_details`` description.
        ``training.load_initial_actor`` uses both to start a new run.

    Raises
    ------
    ValueError
        The file fails ``read_portable_model``'s checks, was made from a
        Policy rather than a built-in actor, or the rebuilt weights do not
        match the original ``actor_digest``.

    Notes
    -----
    Host-only; places the arrays on JAX's default device. No program is
    loaded or run.
    """
    source = Path(path)
    portable = read_portable_model(source)
    layout = portable.actor_details.get("actor_layout")
    if portable.model.get("method") not in _TEACHER_KINDS or not isinstance(
        layout, list
    ):
        raise ValueError(
            f"{source} was not made from a built-in actor, so it cannot start training"
        )
    _, parts = _read_parts(source)
    rows = cast(list[dict[str, Any]], portable.model["weights"])
    leaves = _arrays(parts["weights.npz"], rows, f"{source}: weights.npz")
    tree = _nested([str(row["path"]) for row in rows], leaves)
    if tree_digest(tree) != portable.actor_details.get("actor_digest"):
        raise ValueError(f"{source}: rebuilt weights do not match the actor digest")
    return tree, portable.actor_details


# ---------------------------------------------------------------------------
# The loader System
# ---------------------------------------------------------------------------


def _key_data(keys: Array) -> Array:
    """Return raw Threefry key data, refusing other random-key implementations."""
    if jax.dtypes.issubdtype(  # pyright: ignore[reportPrivateImportUsage]
        keys.dtype, jax.dtypes.prng_key
    ):
        if str(jax.random.key_impl(keys)) != "threefry2x32":
            raise ValueError("Portable models require Threefry random keys.")
        return jax.random.key_data(keys)
    if keys.dtype != jnp.uint32 or keys.shape[-1:] != (2,):
        raise TypeError("Keys must be typed Threefry keys or uint32 (..., 2) key data.")
    return keys


def _named_leaves(tree: PolicyTree) -> list[tuple[str, PolicyTree]]:
    """Return (slash-joined path, leaf) pairs in JAX's flatten order."""
    flat = cast(
        list[tuple[tuple[Any, ...], PolicyTree]],
        jax.tree_util.tree_flatten_with_path(tree)[0],
    )
    return [
        (jax.tree_util.keystr(key, simple=True, separator="/"), leaf)
        for key, leaf in flat
    ]


def _all_batched(tree: PolicyTree) -> PolicyTree:
    """Mark every output leaf of a vmap rule as batched."""

    def batched(leaf: Array) -> bool:
        """Say that this output leaf has the vmapped axis."""
        del leaf
        return True

    return jax.tree.map(batched, tree)


def _leading_one(leaf: Array) -> Array:
    """Add a leading axis of one to an array."""
    return leaf[None]


def _flat_rows(
    tree: PolicyTree, rows: Sequence[Mapping[str, Any]], what: str
) -> list[Array]:
    """Flatten one team's (B,5,...) input tree to (B*5,...) leaves in file order.

    The leaf paths, per-agent shapes and dtypes must equal the file's record;
    otherwise the input layout changed since the file was made, and the call
    is refused with ValueError (at trace time; no device work is wasted).
    """
    found = _named_leaves(tree)
    expected = [
        (str(row["path"]), tuple(row["shape"]), str(row["dtype"])) for row in rows
    ]
    actual = [(name, tuple(leaf.shape[2:]), str(leaf.dtype)) for name, leaf in found]
    if actual != expected:
        raise ValueError(
            f"This portable model expects a different {what} layout than this "
            "MARL-BGs provides; load it with the MARL-BGs version that made it."
        )
    return [
        leaf.reshape((leaf.shape[0] * leaf.shape[1], *leaf.shape[2:]))
        for _, leaf in found
    ]


def _vmappable(call: Callable[..., Any]) -> Callable[..., Any]:
    """Make a deserialized program usable under jax.vmap.

    A deserialized exported call has no batching rule of its own. Its rows
    are independent, so a vmapped axis is folded into the row axis and the
    program runs once. When the weights themselves arrive batched (for
    example vmap over a population, or lax.cond inside vmap), the rule loops
    over the batch with lax.map instead: correct, slower, never an error.
    Nested vmap works because the rule calls the wrapped function again.
    """

    @jax.custom_batching.custom_vmap
    def wrapped(weights: PolicyTree, *rows: PolicyTree) -> PolicyTree:
        """Run the program once on unbatched weights and R agent rows."""
        return call(weights, *rows)

    @wrapped.def_vmap
    def rule(  # pyright: ignore[reportUnusedFunction]
        axis_size: int, in_batched: PolicyTree, weights: PolicyTree, *rows: PolicyTree
    ) -> PolicyTree:
        """Fold the vmapped axis into the rows, or loop when weights are batched."""

        def full(value: Array, batched: bool) -> Array:
            """Give an unbatched leaf the vmapped axis."""
            return (
                value if batched else jnp.broadcast_to(value, (axis_size, *value.shape))
            )

        if any(jax.tree.leaves(in_batched[0])):
            args = jax.tree.map(full, (weights, *rows), tuple(in_batched))

            def each(one: tuple[PolicyTree, ...]) -> PolicyTree:
                """Run the program for one element of the batch."""
                return wrapped(*one)

            out = cast(PolicyTree, jax.lax.map(each, args))
            return out, _all_batched(out)

        def fold(value: Array, batched: bool) -> Array:
            """Merge the vmapped axis into the row axis."""
            whole = full(value, batched)
            return whole.reshape((whole.shape[0] * whole.shape[1], *whole.shape[2:]))

        def unfold(value: Array) -> Array:
            """Split the row axis back into the vmapped axis and the rows."""
            return value.reshape(
                (axis_size, value.shape[0] // axis_size, *value.shape[1:])
            )

        out = jax.tree.map(
            unfold, wrapped(weights, *jax.tree.map(fold, rows, tuple(in_batched[1:])))
        )
        return out, _all_batched(out)

    return wrapped


@dataclass(frozen=True)
class _Program:
    """One deserialized decision program and the layout it was exported with."""

    call: Callable[..., Any]
    actor_rows: tuple[dict[str, Any], ...]
    mask_rows: tuple[dict[str, Any], ...]
    memory_rows: tuple[dict[str, Any], ...]


def _portable_apply(
    program: _Program,
    variables: PortableVariables,
    memory: PolicyTree,
    inputs: SystemInput,
    keys: Array,
) -> SystemOutput:
    """Run one portable decision for every slot of every game, as the original did.

    Agent i of game b gets the key ``split(keys[b], 5)[i]``, the built-in
    learners' rule. Memory resets to the starting memory on a valid episode
    start, invalid lanes keep their memory unchanged, and every slot is
    computed whether or not this System controls it. Actions are int32
    (B,5); learning outputs are the teacher outputs, shaped (B,5,K).
    """
    games = inputs.valid.shape[0]
    agent_keys = jax.vmap(functools.partial(jax.random.split, num=_TEAM_SLOTS))(keys)
    key_data = _key_data(agent_keys).reshape(games * _TEAM_SLOTS, 2)
    actors = _flat_rows(inputs.actors, program.actor_rows, "actor input")
    masks = _flat_rows(inputs.action_mask, program.mask_rows, "action mask")
    old = tuple(jax.tree.leaves(memory))
    start = (inputs.episode_start & inputs.valid)[:, None]
    begin = tuple(
        jnp.where(
            start.reshape(start.shape + (1,) * (leaf.ndim - 2)),
            jnp.broadcast_to(first, leaf.shape),
            leaf,
        ).reshape((leaf.shape[0] * leaf.shape[1], *leaf.shape[2:]))
        for leaf, first in zip(old, variables.initial_memory, strict=True)
    )
    move, target, ultimate, after, teacher = program.call(
        tuple(variables.weights), begin, tuple(actors), tuple(masks), key_data
    )
    valid = inputs.valid[:, None]
    kept = tuple(
        jnp.where(
            valid.reshape(valid.shape + (1,) * (leaf.ndim - 2)),
            new.reshape(leaf.shape),
            leaf,
        )
        for leaf, new in zip(old, after, strict=True)
    )
    shape = (games, _TEAM_SLOTS)
    return SystemOutput(
        ActorAction(
            move.reshape(shape), target.reshape(shape), ultimate.reshape(shape)
        ),
        jax.tree.unflatten(cast(Any, jax.tree.structure(memory)), kept),
        learning_outputs=teacher.reshape((*shape, teacher.shape[-1])),
    )


def _portable_init(variables: PortableVariables, inputs: SystemInput) -> PolicyTree:
    """Return fresh (B,5,...) memory: each agent's starting memory, per slot."""
    leaves = tuple(
        jnp.broadcast_to(first, (*inputs.active_mask.shape, *first.shape))
        for first in variables.initial_memory
    )
    if not leaves:
        return ()
    return leaves[0] if len(leaves) == 1 else leaves


_PROGRAMS: dict[str, tuple[Any, Any]] = {}


def _program(
    blob_sha256: str, blob: bytes, model: Mapping[str, Any]
) -> tuple[Any, Any]:
    """Deserialize one program once and build its two System hooks.

    Hooks are cached by the program's SHA-256, so every save of one model
    shares one apply and init object and therefore one compiled program. The
    input and memory layout is part of the program, so the first file's
    record serves all. The hooks are tiny and fixed: saved runs record their
    code digest, and a test pins it. Raises ValueError when the program turns
    off any of JAX's own safety checks (for example its shape assertions).
    """
    if blob_sha256 in _PROGRAMS:
        return _PROGRAMS[blob_sha256]
    from jax import export

    exported = export.deserialize(bytearray(blob))
    if exported.disabled_safety_checks:
        raise ValueError(
            "A portable model program must not turn off JAX's safety checks"
        )
    program = _Program(
        _vmappable(cast(Callable[..., PolicyTree], exported.call)),
        tuple(model["inputs"]["actor_input"]),
        tuple(model["inputs"]["action_mask"]),
        tuple(model["memory"]),
    )

    def apply(
        variables: PortableVariables,
        memory: PolicyTree,
        inputs: SystemInput,
        keys: Array,
    ) -> SystemOutput:
        """Choose the team's actions with the portable program (see _portable_apply)."""
        return _portable_apply(program, variables, memory, inputs, keys)

    def init(
        variables: PortableVariables, inputs: SystemInput, keys: Array
    ) -> PolicyTree:
        """Return fresh per-slot memory (see _portable_init)."""
        del keys
        return _portable_init(variables, inputs)

    _PROGRAMS[blob_sha256] = (_declare_lane_independent(apply), init)
    return _PROGRAMS[blob_sha256]


def load_portable_model(path: str | Path) -> System:
    """Load a portable model file as an ordinary JAX System for one team.

    Parameters
    ----------
    path : str or Path
        A portable model file (``.zip``), for example a downloaded released
        save. Released names are resolved by ``marl_bgs.load_method``.

    Returns
    -------
    System
        A JAX System that runs the file's program once per agent for all B x 5
        slots. Its name is the file's display name; ``checkpoint`` is the
        original actor's inference digest; its one ``components`` entry
        records the format and step (``version``), the file's SHA-256
        (``checkpoint``), the program's SHA-256 (``code_digest``) and the
        weights' digest (``parameters_digest``). Learning outputs are the
        teacher outputs, float32 (B,5,198) for the built-in families.

    Raises
    ------
    ValueError
        The file fails ``read_portable_model``'s checks, its arrays are
        compressed or do not match ``model.json``, or its program turns off
        JAX's safety checks. A later call refuses inputs whose layout differs
        from the file's record, and non-Threefry typed keys.
    OSError
        The file cannot be read.

    Notes
    -----
    Host-only setup; no program runs here. Loading needs the ``flatbuffers``
    package but not the training extra. Agent i of game b uses the key
    ``split(keys[b], 5)[i]``, as the built-in learners do. Memory resets on a
    valid episode start; invalid lanes keep their memory. The apply hook is
    declared lane-independent, so a pool runs it only on its own games. Saves
    of one model share one program and one compiled function; weights stay
    dynamic. Under a GPU's normal (non-deterministic) mode, games can differ
    slightly from the original actor's recorded games; on CPU or in
    deterministic mode the same file and inputs replay the same games.

    Examples
    --------
    >>> import marl_battlegrounds as marl_bgs
    >>> system = marl_bgs.load_method("step-250_675_200.zip")  # doctest: +SKIP
    """
    source = Path(path)
    portable = read_portable_model(source)
    _, parts = _read_parts(source)
    model = portable.model
    weights = _arrays(parts["weights.npz"], model["weights"], f"{source}: weights.npz")
    memory = _arrays(
        parts["initial_memory.npz"], model["memory"], f"{source}: initial_memory.npz"
    )
    blob = parts["decide.jaxexport"]
    blob_sha256 = model["parts"]["decide.jaxexport"]
    apply, init = _program(blob_sha256, blob, model)
    step = model.get("training_step")
    return System(
        str(model.get("display_name") or source.stem),
        apply,
        variables=PortableVariables(weights, memory),
        init=init,
        components=(
            {
                "name": str(model.get("display_name") or source.stem),
                "version": f"{PORTABLE_MODEL_FORMAT}-{PORTABLE_MODEL_FORMAT_VERSION}"
                + (f"@{step}" if step is not None else ""),
                "checkpoint": portable.sha256,
                "code_digest": blob_sha256,
                "parameters_digest": str(model.get("actor_digest")),
            },
        ),
        checkpoint=model.get("inference_digest"),
    )


# ---------------------------------------------------------------------------
# Per-agent decision functions for the built-in families
# ---------------------------------------------------------------------------


def _world_vector(values: Array, flag: Array | None) -> Array:
    """Map one 198-wide vector from the reflected frame back to the world frame.

    values is (198,) in the frame the agent decided in; flag is a 0-d Boolean
    (True when that frame was reflected) or None in the world frame. World
    action j is reflected action mirror(j), so entry j reads entry mirror(j).
    """
    if flag is None:
        return values
    from marl_battlegrounds.baselines.actions import mirror_action_indices

    index = jnp.arange(_NUM_ACTIONS, dtype=jnp.int32)
    return values[mirror_action_indices(index, jnp.broadcast_to(flag, index.shape))]


def _agent_view(
    actor: ActorInput,
    mask: ActionMask,
    frame: str,
    schema: int,
) -> tuple[Array, Array, Array | None]:
    """Encode one agent's permitted view in its training frame.

    actor and mask carry one agent with a leading axis of one. Returns the
    encoded features (1,F), the categorical mask (1,198) and the reflection
    flag (1,) or None in the world frame. Obstacle partners are computed from
    the agent's own table, which equals the shared per-game table.
    """
    from marl_battlegrounds.baselines.actions import categorical_action_mask
    from marl_battlegrounds.baselines.inputs import (
        encode_actor_inputs,
        schema_1_actor_input,
        spawn_frame_flag,
    )
    from marl_battlegrounds.policies.input import mirror_team_view

    if schema == 1:
        actor = schema_1_actor_input(actor)
    flag = None
    if frame != "world":
        flag = spawn_frame_flag(actor, frame)
        actor, mask = mirror_team_view(actor, mask, flag)
    return encode_actor_inputs(actor), categorical_action_mask(mask), flag


def _finish(
    index: Array, teacher: Array, flag: Array | None
) -> tuple[ActorAction, Array]:
    """Map a chosen frame index and its teacher vector to world-frame outputs."""
    from marl_battlegrounds.baselines.actions import (
        decode_actions,
        mirror_action_indices,
    )

    world = index if flag is None else mirror_action_indices(index, flag)
    action = decode_actions(world)
    return (
        ActorAction(action.move[0], action.select_target[0], action.use_ultimate[0]),
        _world_vector(teacher[0], None if flag is None else flag[0]),
    )


class _PQNRow:
    """Build PQN-VDN's per-agent network: PQNNetwork's shared path, any slot count."""

    @staticmethod
    @functools.cache
    def module(input_scale: float, normalization: str) -> PolicyTree:
        """Return a Flax module with PQNNetwork's parameter names and math.

        PQNNetwork reshapes to exactly five slots; this copy reads the slot
        count from its inputs so one agent can run alone. Layers, names,
        constants and order are the same, so the saved variables apply as is.
        """
        import flax.linen as nn

        from marl_battlegrounds.baselines.pqn import (
            _BATCH_NORM_EPSILON,  # pyright: ignore[reportPrivateUsage]
            _BATCH_NORM_MOMENTUM,  # pyright: ignore[reportPrivateUsage]
            _LAYER_NORM_EPSILON,  # pyright: ignore[reportPrivateUsage]
            PQN_HIDDEN_SIZE,
            PQNScannedRNN,
        )

        class PQNAgentNetwork(nn.Module):
            """PQNNetwork's inference path with the slot count taken from active."""

            @nn.compact
            def __call__(
                self,
                memory: Array,
                features: Array,
                episode_start: Array,
                valid: Array,
                active: Array,
            ) -> tuple[Array, Array]:
                """Return memory (games,slots,512) and values (rows,games,slots,198)."""
                rows, games = valid.shape
                slots = active.shape[-1]
                actors = games * slots
                if input_scale != 1.0:
                    features = features * input_scale
                row = valid[..., None] & active
                x = jnp.where(row[..., None], features, 0.0).reshape(rows, actors, -1)

                def normalize(value: Array) -> Array:
                    """Apply one BatchNorm with the saved running statistics."""
                    return nn.BatchNorm(
                        use_running_average=True,
                        momentum=_BATCH_NORM_MOMENTUM,
                        epsilon=_BATCH_NORM_EPSILON,
                    )(value, mask=None)

                x = normalize(x)
                for _ in range(2):
                    hidden = nn.Dense(PQN_HIDDEN_SIZE)(x)
                    x = nn.relu(
                        nn.LayerNorm(epsilon=_LAYER_NORM_EPSILON)(hidden)
                        if normalization == "layer_norm"
                        else normalize(hidden)
                    )

                def per_actor(flags: Array) -> Array:
                    """Broadcast (rows,games) flags to (rows,games*slots) actor rows."""
                    return jnp.broadcast_to(
                        flags[..., None], (rows, games, slots)
                    ).reshape(rows, actors)

                carry, x = PQNScannedRNN(name="ScannedRNN_0")(
                    memory.reshape(actors, PQN_HIDDEN_SIZE),
                    (
                        x,
                        per_actor(episode_start),
                        per_actor(valid),
                        active.reshape(rows, actors),
                    ),
                )
                values = nn.Dense(_NUM_ACTIONS)(x).reshape(
                    rows, games, slots, _NUM_ACTIONS
                )
                values = jnp.where(active[..., None], values, 0.0)
                return carry.reshape(games, slots, PQN_HIDDEN_SIZE), values

        return PQNAgentNetwork()


type _Decide = Callable[
    [Any, tuple[Array, ...], Any, Any, Array],
    tuple[ActorAction, tuple[Array, ...], Array],
]


def _builtin_decide(method: str, settings: Mapping[str, Any]) -> _Decide:
    """Return one agent's decision function for a built-in actor family.

    The function takes (variables, memory leaves, one agent's ActorInput, its
    ActionMask, a typed Threefry key) without batch axes and returns the
    action, the next memory leaves and the 198 teacher outputs in the world
    frame. Episode resets and invalid lanes are the loader's job.
    """
    scale = float(settings["input_scale"])
    frame = str(settings["spawn_frame"])
    schema = int(settings["actor_input_schema"])

    def one(tree: PolicyTree) -> PolicyTree:
        """Give every leaf a leading axis of one."""
        return jax.tree.map(_leading_one, tree)

    off = jnp.zeros((1, 1, 1), jnp.bool_)
    on = jnp.ones((1, 1, 1), jnp.bool_)

    if method in ("mappo", "ippo", "ff_mappo", "ff_ippo"):
        from marl_battlegrounds.baselines.actions import (
            _log_probabilities,  # pyright: ignore[reportPrivateUsage]
            sample_actions,
        )
        from marl_battlegrounds.baselines.ppo import FeedForwardActor, RecurrentActor

        recurrent = method in ("mappo", "ippo")

        def ppo(
            variables: PolicyTree,
            memory: tuple[Array, ...],
            actor: ActorInput,
            mask: ActionMask,
            key: Array,
        ) -> tuple[ActorAction, tuple[Array, ...], Array]:
            """Sample one PPO action and report its log-probabilities."""
            features, legal, flag = _agent_view(one(actor), one(mask), frame, schema)
            if recurrent:
                carry, logits = cast(
                    tuple[Array, Array],
                    RecurrentActor(input_scale=scale).apply(
                        variables, memory[0][None, None], features[None, None], off, on
                    ),
                )
                logits, after = logits[0, 0], (carry[0, 0],)
            else:
                logits = cast(
                    Array,
                    FeedForwardActor(input_scale=scale).apply(variables, features),
                )
                after = ()
            index = sample_actions(logits, legal, key[None])
            action, teacher = _finish(index, _log_probabilities(logits, legal), flag)
            return action, after, teacher

        return ppo

    if method == "qmix":
        from marl_battlegrounds.baselines.qmix import RecurrentQNetwork, greedy_actions

        def qmix(
            variables: PolicyTree,
            memory: tuple[Array, ...],
            actor: ActorInput,
            mask: ActionMask,
            key: Array,
        ) -> tuple[ActorAction, tuple[Array, ...], Array]:
            """Choose one greedy QMIX action and report its action values."""
            del key
            features, legal, flag = _agent_view(one(actor), one(mask), frame, schema)
            carry, values = cast(
                tuple[Array, Array],
                RecurrentQNetwork(input_scale=scale).apply(
                    variables, memory[0][None, None], features[None, None], off, on
                ),
            )
            values = values[0, 0]
            index = greedy_actions(values, legal)
            teacher = jnp.where(legal, values, -jnp.inf)
            action, world = _finish(index, teacher, flag)
            return action, (carry[0, 0],), world

        return qmix

    if method == "pqn_vdn":
        from marl_battlegrounds.baselines.qmix import greedy_actions
        from marl_battlegrounds.core.types import AGENT_FEATURE_CLASS_ID

        network = _PQNRow.module(scale, str(settings["normalization"]))

        def pqn(
            variables: PolicyTree,
            memory: tuple[Array, ...],
            actor: ActorInput,
            mask: ActionMask,
            key: Array,
        ) -> tuple[ActorAction, tuple[Array, ...], Array]:
            """Choose one greedy PQN-VDN action; an unused slot gets zero memory."""
            del key
            # A roster member (alive or dead) has a class; an unused slot's view is
            # all zero.
            active = (actor.observation.self_features[AGENT_FEATURE_CLASS_ID] > 0)[
                None, None, None
            ]
            features, legal, flag = _agent_view(one(actor), one(mask), frame, schema)
            carry, values = cast(
                tuple[Array, Array],
                network.apply(
                    variables,
                    memory[0][None, None],
                    features[None, None],
                    jnp.zeros((1, 1), jnp.bool_),
                    jnp.ones((1, 1), jnp.bool_),
                    active,
                ),
            )
            values = values[0, 0]
            index = greedy_actions(values, legal)
            teacher = jnp.where(legal, values, -jnp.inf)
            action, world = _finish(index, teacher, flag)
            return action, (carry[0, 0],), world

        return pqn

    raise ValueError(f"Unknown built-in actor method {method!r}")


# ---------------------------------------------------------------------------
# Exporting
# ---------------------------------------------------------------------------


@functools.cache
def _input_rows() -> tuple[
    tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], Any, Any
]:
    """Describe one agent's ActorInput and ActionMask leaves from a real game.

    Returns the actor-input and action-mask row descriptions (path, per-agent
    shape, dtype) and the two tree structures. Uses jax.eval_shape, so no
    game is simulated.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.policy_execution import system_inputs

    env = marl_bgs.make("tdm", map_id=0, num_envs=2, metrics="none")

    def prepare(key: Array) -> SystemInput:
        """Build one team's inputs for one game."""
        observations, state = env.reset(key)
        return system_inputs(observations, state, team=0)

    inputs = jax.eval_shape(prepare, jax.random.key(0))

    def rows(tree: PolicyTree) -> tuple[dict[str, Any], ...]:
        """List each leaf's path, per-agent shape and dtype."""
        return tuple(
            {
                "path": name,
                "shape": list(leaf.shape[2:]),
                "dtype": str(leaf.dtype),
            }
            for name, leaf in _named_leaves(tree)
        )

    return (
        rows(inputs.actors),
        rows(inputs.action_mask),
        cast(Any, jax.tree.structure(inputs.actors)),
        cast(Any, jax.tree.structure(inputs.action_mask)),
    )


def _leaf_rows(tree: PolicyTree) -> list[dict[str, Any]]:
    """List a tree's leaves as path, shape and dtype rows in flatten order."""
    return [
        {
            "path": name,
            "shape": list(np.shape(leaf)),
            "dtype": str(np.asarray(leaf).dtype),
        }
        for name, leaf in _named_leaves(tree)
    ]


def _export_program(
    decide: _Decide,
    weights: PolicyTree,
    memory: PolicyTree,
    teacher_width: int,
) -> jax_export.Exported:
    """Export decide over R independent rows with a symbolic R for every platform."""
    from jax import export

    actor_rows, mask_rows, actor_def, mask_def = _input_rows()
    weight_leaves, weight_def = cast(tuple[list[Any], Any], jax.tree.flatten(weights))
    memory_leaves, memory_def = cast(tuple[list[Any], Any], jax.tree.flatten(memory))

    def rows(
        flat_weights: tuple[Array, ...],
        flat_memory: tuple[Array, ...],
        actor_leaves: tuple[Array, ...],
        mask_leaves: tuple[Array, ...],
        key_data: Array,
    ) -> tuple[Array, Array, Array, tuple[Array, ...], Array]:
        """Run decide on every row; rows never see each other."""
        variables = jax.tree.unflatten(weight_def, list(flat_weights))
        actors = jax.tree.unflatten(actor_def, list(actor_leaves))
        masks = jax.tree.unflatten(mask_def, list(mask_leaves))
        keys = cast(Array, jax.random.wrap_key_data(key_data, impl="threefry2x32"))

        def agent(
            carry: tuple[Array, ...],
            actor: ActorInput,
            mask: ActionMask,
            key: Array,
        ) -> tuple[Array, Array, Array, tuple[Array, ...], Array]:
            """Decide for one agent and return flat outputs."""
            leaves = jax.tree.leaves(jax.tree.unflatten(memory_def, list(carry)))
            action, after, teacher = decide(variables, tuple(leaves), actor, mask, key)
            if teacher_width == 0:
                teacher = jnp.zeros((0,), jnp.float32)
            return (
                action.move,
                action.select_target,
                action.use_ultimate,
                tuple(jax.tree.leaves(after)),
                teacher.astype(jnp.float32),
            )

        return jax.vmap(agent)(flat_memory, actors, masks, keys)

    (count,) = export.symbolic_shape("r")

    def spec(shape: Sequence[int], dtype: str) -> jax.ShapeDtypeStruct:
        """Give one per-agent leaf the symbolic row axis."""
        return jax.ShapeDtypeStruct((count, *shape), jnp.dtype(dtype))

    # Source locations would put this machine's file paths and line numbers
    # into the program and make two exports of one model differ. Leave them out.
    limit = cast(
        int,
        jax.config.jax_traceback_in_locations_limit,  # pyright: ignore[reportAttributeAccessIssue]
    )
    jax.config.update("jax_traceback_in_locations_limit", 0)
    try:
        return export.export(jax.jit(rows), platforms=list(PLATFORMS))(
            tuple(
                jax.ShapeDtypeStruct(np.shape(leaf), np.asarray(leaf).dtype)
                for leaf in weight_leaves
            ),
            tuple(
                spec(np.shape(leaf), str(np.asarray(leaf).dtype))
                for leaf in memory_leaves
            ),
            tuple(spec(row["shape"], row["dtype"]) for row in actor_rows),
            tuple(spec(row["shape"], row["dtype"]) for row in mask_rows),
            jax.ShapeDtypeStruct((count, 2), jnp.uint32),
        )
    finally:
        jax.config.update("jax_traceback_in_locations_limit", limit)


def _npz(rows: Sequence[Mapping[str, Any]], leaves: Sequence[Any]) -> bytes:
    """Write named arrays to uncompressed .npz bytes in a fixed order."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for row, leaf in zip(rows, leaves, strict=True):
            array = io.BytesIO()
            np.lib.format.write_array(array, np.asarray(leaf), allow_pickle=False)
            info = zipfile.ZipInfo(f"{row['path']}.npy", _ZIP_TIME)
            archive.writestr(info, array.getvalue())
    return buffer.getvalue()


def _write(path: Path, model: dict[str, Any], parts: dict[str, bytes]) -> Path:
    """Write the five parts into one uncompressed ZIP with fixed timestamps."""
    model = {
        **model,
        "parts": {name: hashlib.sha256(parts[name]).hexdigest() for name in _PARTS[1:]},
    }
    data = {
        "model.json": json.dumps(model, indent=2, sort_keys=True).encode() + b"\n",
        **parts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
        for name in _PARTS:
            archive.writestr(zipfile.ZipInfo(name, _ZIP_TIME), data[name])
    temporary.replace(path)
    return path


def _base_model(display_name: str) -> dict[str, Any]:
    """Return the model.json fields every portable file shares."""
    actor_rows, mask_rows, _, _ = _input_rows()
    return {
        "format": PORTABLE_MODEL_FORMAT,
        "format_version": PORTABLE_MODEL_FORMAT_VERSION,
        "game_mode": "tdm",
        "display_name": display_name,
        "jax_version": jax.__version__,
        "threefry_partitionable": bool(
            jax.config.jax_threefry_partitionable  # pyright: ignore[reportAttributeAccessIssue, reportUnknownArgumentType]
        ),
        "platforms": list(PLATFORMS),
        "rows": "r",
        "inputs": {
            "actor_input": list(actor_rows),
            "action_mask": list(mask_rows),
            "key_data": {"shape": [2], "dtype": "uint32", "impl": "threefry2x32"},
        },
        "action_codes": {"version": 1, "count": _NUM_ACTIONS},
        "message_shape": [],
    }


# Serialized programs of built-in actors, keyed by family, settings and layout.
_BUILTIN_PROGRAMS: dict[str, bytes] = {}


def _export_builtin(folder: Path, path: Path) -> Path:
    """Export a saved built-in actor folder (see export_model)."""
    from marl_battlegrounds.training._checkpoint_metadata import (
        _actor_input_scale,  # pyright: ignore[reportPrivateUsage]
        _actor_normalization,  # pyright: ignore[reportPrivateUsage]
        _actor_parameter_sharing,  # pyright: ignore[reportPrivateUsage]
        _actor_spawn_frame,  # pyright: ignore[reportPrivateUsage]
        _inference_digest,  # pyright: ignore[reportPrivateUsage]
        _schema_method,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training.checkpoints import (
        _actor_template,  # pyright: ignore[reportPrivateUsage]
        _layout,  # pyright: ignore[reportPrivateUsage]
        _restore_arrays,  # pyright: ignore[reportPrivateUsage]
        read_checkpoint_details,
    )

    details = read_checkpoint_details(folder)
    if details.get("kind") != "actor":
        raise ValueError(f"{folder} is not a saved actor folder")
    method = _schema_method(details["schemas"])
    sharing = _actor_parameter_sharing(details)
    if sharing != "all":
        raise ValueError(
            f"{folder} was trained with parameter_sharing={sharing!r}; format 1 "
            "exports only actors whose agents share one set of weights"
        )
    settings = {
        "input_scale": _actor_input_scale(details),
        "spawn_frame": _actor_spawn_frame(details),
        "actor_input_schema": int(details["schemas"]["actor_input"]),
        "normalization": _actor_normalization(details),
    }
    template = _actor_template(
        method, settings["actor_input_schema"], sharing, settings["normalization"]
    )
    if details["actor_layout"] != _layout(template):
        raise ValueError(f"{folder}: its weights do not match the installed model")
    actor = _restore_arrays(folder / "actor", template, None)
    if tree_digest(actor) != details["actor_digest"]:
        raise ValueError(f"{folder}: restored weights do not match the actor digest")
    hidden = {"mappo": 128, "ippo": 128, "qmix": 256, "pqn_vdn": 512}.get(method)
    memory = {"memory": np.zeros((hidden,), np.float32)} if hidden else {}
    # The program depends only on the family, its settings and the weight
    # layout, so every save of one model shares it; export it once.
    program_key = json.dumps(
        [method, settings, details["actor_layout"]], sort_keys=True
    )
    if program_key not in _BUILTIN_PROGRAMS:
        decide = _builtin_decide(method, settings)
        exported = _export_program(decide, actor, memory, _NUM_ACTIONS)
        _BUILTIN_PROGRAMS[program_key] = bytes(exported.serialize())
    metadata = details.get("metadata", {})
    display = metadata.get("display_name") or method
    raw = (folder / "actor_details.json").read_bytes()
    model = {
        **_base_model(str(display)),
        "method": method,
        "training_step": metadata.get("env_steps"),
        "checkpoint_id": details.get("checkpoint_id"),
        "inference_digest": _inference_digest(details),
        "actor_digest": details["actor_digest"],
        "actor_input_schema": settings["actor_input_schema"],
        "input_scale": settings["input_scale"],
        "spawn_frame": settings["spawn_frame"],
        "weights": _leaf_rows(actor),
        "memory": _leaf_rows(memory),
        "outputs": {
            "action": ["move", "select_target", "use_ultimate"],
            "teacher_outputs": {
                "kind": _TEACHER_KINDS[method],
                "shape": [_NUM_ACTIONS],
                "frame": "world",
                "illegal_actions": "-inf",
            },
        },
    }
    return _write(
        path,
        model,
        {
            "decide.jaxexport": _BUILTIN_PROGRAMS[program_key],
            "weights.npz": _npz(_leaf_rows(actor), jax.tree.leaves(actor)),
            "initial_memory.npz": _npz(_leaf_rows(memory), jax.tree.leaves(memory)),
            "actor_details.json": raw,
        },
    )


def _export_policy(policy: Policy, path: Path) -> Path:
    """Export a per-agent JAX Policy (see export_model)."""
    if policy.execution != "jax":
        raise ValueError(
            f"Policy {policy.name!r} runs on the host; "
            "only JAX Policies can be exported"
        )

    def decide(
        variables: PolicyTree,
        memory: tuple[Array, ...],
        actor: ActorInput,
        mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, tuple[Array, ...], Array]:
        """Call the Policy's own apply for one agent."""
        carry = jax.tree.unflatten(
            cast(Any, jax.tree.structure(policy.initial_carry)), list(memory)
        )
        action, after = policy.apply(variables, carry, actor, mask, key)
        return action, tuple(jax.tree.leaves(after)), jnp.zeros((0,), jnp.float32)

    exported = _export_program(decide, policy.variables, policy.initial_carry, 0)
    description = {
        "kind": "policy",
        "name": policy.name,
        "checkpoint": policy.checkpoint,
    }
    model = {
        **_base_model(policy.name),
        "method": "policy",
        "training_step": None,
        "checkpoint_id": policy.checkpoint,
        "inference_digest": policy.checkpoint,
        "actor_digest": tree_digest(policy.variables),
        "weights": _leaf_rows(policy.variables),
        "memory": _leaf_rows(policy.initial_carry),
        "outputs": {
            "action": ["move", "select_target", "use_ultimate"],
            "teacher_outputs": None,
        },
    }
    return _write(
        path,
        model,
        {
            "decide.jaxexport": bytes(exported.serialize()),
            "weights.npz": _npz(
                _leaf_rows(policy.variables), jax.tree.leaves(policy.variables)
            ),
            "initial_memory.npz": _npz(
                _leaf_rows(policy.initial_carry), jax.tree.leaves(policy.initial_carry)
            ),
            "actor_details.json": json.dumps(description, sort_keys=True).encode()
            + b"\n",
        },
    )


def export_model(source: Policy | str | Path, path: str | Path) -> Path:
    """Write one portable model file from a saved actor or a per-agent Policy.

    Parameters
    ----------
    source : Policy, str or Path
        Either a saved built-in actor folder (recurrent or feedforward MAPPO
        or IPPO, QMIX or PQN-VDN) or a JAX ``Policy``: a per-agent function
        ``apply(variables, carry, actor_input, action_mask, key)`` with fixed
        memory shapes. A Policy's ``variables`` and ``initial_carry`` become
        the file's weights and starting memory.
    path : str or Path
        The file to write, normally ending in ``.zip``. Parent folders are
        created. An existing file is replaced.

    Returns
    -------
    Path
        ``path`` as a Path, after the file is complete.

    Raises
    ------
    ValueError
        The folder is not a saved actor, its weights or digest do not match,
        it was trained with ``parameter_sharing`` other than ``"all"`` (format
        1 has no per-slot or per-class weights), or the Policy runs on the
        host (for example a language-model team).
    TypeError
        source is neither a Policy nor a folder path.
    OSError
        Reading the source or writing the file failed.

    Notes
    -----
    Host-only. Exports one program for cpu, cuda, rocm and tpu with an open
    number of agent rows, so the same file serves any batch size. Built-in
    actors also get teacher outputs (198 log-probabilities or action values in
    the world frame). Exporting the same source twice gives identical bytes.
    Exporting a built-in actor folder needs the training extra. Saves of one
    built-in model share one program, so the process exports it once and
    reuses it for later saves. Files are written to a temporary name first
    and then renamed.

    Examples
    --------
    >>> import marl_battlegrounds as marl_bgs
    >>> from marl_battlegrounds import training
    >>> run = training.load_run("runs/my-run")  # doctest: +SKIP
    >>> marl_bgs.export_model(run.final_actor, "my-model.zip")  # doctest: +SKIP
    """
    target = Path(path)
    if isinstance(source, Policy):
        return _export_policy(source, target)
    if isinstance(cast(object, source), (str, Path)):
        return _export_builtin(Path(source), target)
    raise TypeError("export_model needs a Policy or a saved actor folder")
