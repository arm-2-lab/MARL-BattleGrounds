"""Build replays of Capture the Flag games.

V5 binds episode context V5 (the Capture the Flag config V3) to frame V4 (the
flag state in each snapshot) and transition V2 (Core's flag facts and the flag
events). Team Deathmatch and neutral games keep replay V4. V5 reuses the V2
envelope's ordering, completion and digest authorities, as V3 and V4 do.
Builders operate on captured records; metrics, simulator execution and file
publication belong to separate modules. Capture imports stay local to
replay_from_packets so replay readers do not import JAX.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, cast

from pydantic import model_validator

from marl_battlegrounds.evaluation._metrics.metrics import (
    CompletionState,
    RolloutFailureOrigin,
)
from marl_battlegrounds.evaluation._recording.validation import (
    _validate_frame_information_regime,  # pyright: ignore[reportPrivateUsage]
    validate_declared_model_tree,
)
from marl_battlegrounds.evaluation._replays.replay import (
    ReplayWrapperMetadataV1,
    RuntimeProvenanceV1,
)
from marl_battlegrounds.evaluation._replays.replay_v2 import (
    ReplayArtifactHeaderV2,
    ReplayArtifactReferenceV2,
    ReplayArtifactV2,
    _build_replay_content,  # pyright: ignore[reportPrivateUsage]
    _frames_from_packets,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV5,
    EvaluationFrameV4,
    EvaluationTransitionV2,
    canonical_digest_sha256,
    canonical_json_bytes,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation._replays.replay_capture import ReplayPackets


class ReplayArtifactHeaderV5(ReplayArtifactHeaderV2):
    """Replay header with exact context V5, for one Capture the Flag game.

    Attributes
    ----------
    schema_version : Literal[5]
        Fixed integer 5.
    context : EvaluationEpisodeContextV5
        Exact EvaluationEpisodeContextV5, with the Capture the Flag config.

    Notes
    -----
    All other identity, count, runtime and wrapper fields follow
    ReplayArtifactHeaderV2, and so do its checks (T+1 frames for T
    transitions, matching context and runtime versions, ordered wrappers).
    A separate frozen wire root, not an alias for an older header.
    """

    # These exact wire roots share validators, not substitutable version values.
    schema_version: Literal[5] = 5  # pyright: ignore[reportIncompatibleVariableOverride]
    context: EvaluationEpisodeContextV5  # pyright: ignore[reportIncompatibleVariableOverride]


class ReplayArtifactV5(ReplayArtifactV2):
    """Replay envelope for one Capture the Flag game.

    Attributes
    ----------
    schema_version : Literal[5]
        Fixed integer 5.
    header : ReplayArtifactHeaderV5
        Exact ReplayArtifactHeaderV5.
    frames : tuple[EvaluationFrameV4, ...]
        Tuple of EvaluationFrameV4, each holding the flag state.
    transitions : tuple[EvaluationTransitionV2, ...]
        Tuple of EvaluationTransitionV2, with Core's flag facts and events.

    Notes
    -----
    Artifact IDs, completion, transitions and digests follow ReplayArtifactV2's
    common contract; each frame is also checked against the recorded
    information regime. This frozen record holds captured evidence, not a
    policy input, metric report or open file.
    """

    schema_version: Literal[5] = 5  # pyright: ignore[reportIncompatibleVariableOverride]
    header: ReplayArtifactHeaderV5  # pyright: ignore[reportIncompatibleVariableOverride]
    frames: tuple[EvaluationFrameV4, ...]  # pyright: ignore[reportIncompatibleVariableOverride]
    transitions: tuple[EvaluationTransitionV2, ...]  # pyright: ignore[reportIncompatibleVariableOverride]

    @model_validator(mode="after")
    def _validate_current_frames(self) -> ReplayArtifactV5:
        """Check every frame against the recorded execution information contract."""
        for frame in self.frames:
            _validate_frame_information_regime(self.header.context, frame)
        return self


class ReplayArtifactReferenceV5(ReplayArtifactReferenceV2):
    """Path-free reference to original replay V5 bytes.

    Attributes
    ----------
    schema_version : Literal[5]
        Fixed reference version 5.
    replay_schema_version : Literal[5]
        Fixed replay version 5.

    Notes
    -----
    IDs, content digests and positive canonical byte length use the shared
    ReplayArtifactReferenceV2 field contract.
    """

    schema_version: Literal[5] = 5  # pyright: ignore[reportIncompatibleVariableOverride]
    replay_schema_version: Literal[5] = 5  # pyright: ignore[reportIncompatibleVariableOverride]


def build_replay_v5(
    context: EvaluationEpisodeContextV5,
    frames: Sequence[EvaluationFrameV4],
    transitions: Sequence[EvaluationTransitionV2],
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    completion_state: CompletionState | None = None,
    end_or_failure_reason: str | None = None,
    failure_origin: RolloutFailureOrigin | None = None,
) -> ReplayArtifactV5:
    """Seal one Capture the Flag game's captured rows into a validated V5 replay.

    Parameters
    ----------
    context : EvaluationEpisodeContextV5
        Exact EvaluationEpisodeContextV5 for this captured episode.
    frames : Sequence[EvaluationFrameV4]
        Nonempty sequence of exact EvaluationFrameV4, contiguous from zero.
    transitions : Sequence[EvaluationTransitionV2]
        EvaluationTransitionV2 sequence with one fewer row than frames.
    runtime_provenance, wrapper_stack, completion_state, end_or_failure_reason,
    failure_origin
        Same meanings and defaults as in build_replay_v4.

    Returns
    -------
    ReplayArtifactV5
        Frozen ReplayArtifactV5 with checked frame and transition joins,
        information regime, completion and canonical digests.

    Raises
    ------
    TypeError
        The context is not exact V5, a frame is not exact V4 or a transition
        is not exact V2.
    ValueError
        Capture counts, identities, information regime, completion or
        recorded metadata are inconsistent.

    Notes
    -----
    Host-only; no game execution, metric calculation or file publication. An
    initial-only partial prefix is allowed.
    """
    if type(context) is not EvaluationEpisodeContextV5:
        raise TypeError("Capture the Flag replays require exact context V5")
    if any(type(frame) is not EvaluationFrameV4 for frame in frames):
        raise TypeError("Capture the Flag replays require exact frame V4")
    if any(type(row) is not EvaluationTransitionV2 for row in transitions):
        raise TypeError("Capture the Flag replays require exact transition V2")
    content = _build_replay_content(
        context,
        tuple(frames),
        tuple(transitions),
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
        completion_state=completion_state,
        end_or_failure_reason=end_or_failure_reason,
        failure_origin=failure_origin,
        header_type=ReplayArtifactHeaderV5,
    )
    payload = {
        "schema_id": "marl_battlegrounds.evaluation.replay_artifact",
        "schema_version": 5,
        "artifact_id": f"{context.identity.episode_id}:replay",
        "trajectory_content_digest_sha256": canonical_digest_sha256(content),
        **content,
    }
    return ReplayArtifactV5.model_validate(
        {**payload, "canonical_digest_sha256": canonical_digest_sha256(payload)}
    )


def replay_reference_v5(replay: ReplayArtifactV5) -> ReplayArtifactReferenceV5:
    """Describe the original canonical bytes of an exact V5 replay.

    Parameters
    ----------
    replay : ReplayArtifactV5
        Exact ReplayArtifactV5.

    Returns
    -------
    ReplayArtifactReferenceV5
        Its identities, digests and canonical JSON byte length; no local path.

    Raises
    ------
    TypeError
        replay is not exact V5.

    Notes
    -----
    Host-only serialization; reads and writes no file.
    """
    if type(replay) is not ReplayArtifactV5:
        raise TypeError("replay V5 references require exact ReplayArtifactV5")
    return ReplayArtifactReferenceV5(
        artifact_id=replay.artifact_id,
        episode_id=replay.header.context.identity.episode_id,
        context_digest_sha256=replay.header.context_digest_sha256,
        trajectory_content_digest_sha256=replay.trajectory_content_digest_sha256,
        canonical_digest_sha256=replay.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(replay)),
    )


def validate_replay_artifact_v5(replay: ReplayArtifactV5) -> None:
    """Validate a V5 replay's exact model tree without converting versions.

    Parameters
    ----------
    replay : ReplayArtifactV5
        Expected exact ReplayArtifactV5 with declared nested model types.

    Returns
    -------
    None
        None when the complete declared tree is valid.

    Raises
    ------
    TypeError
        The root or a nested model has the wrong exact type.
    ValueError
        Stored fields violate replay or model invariants.

    Notes
    -----
    Host-only and read-only; it does not repair or rewrite a replay.
    """
    validate_declared_model_tree(
        replay, record_name="replay", expected_type=ReplayArtifactV5
    )


def replay_from_packets(
    context: EvaluationEpisodeContextV5,
    packets: Iterable[ReplayPackets],
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    completion_state: CompletionState | None = None,
    end_or_failure_reason: str | None = None,
    failure_origin: RolloutFailureOrigin | None = None,
) -> ReplayArtifactV5:
    """Build a Capture the Flag replay from one ordered stream of scalar packets.

    Parameters
    ----------
    context : EvaluationEpisodeContextV5
        Exact EvaluationEpisodeContextV5 matching the initial packet config.
    packets : Iterable[ReplayPackets]
        Valid scalar ReplayPackets for one positive episode ID, starting at
        zero with initial=True, with consecutive indexes and nothing after
        completion.
    runtime_provenance, wrapper_stack, completion_state, end_or_failure_reason,
    failure_origin
        Same meanings and defaults as in build_replay_v5.

    Returns
    -------
    ReplayArtifactV5
        Validated ReplayArtifactV5 with frame V4 and transition V2 records.

    Raises
    ------
    TypeError
        The context or captured records have incompatible exact types.
    ValueError
        The stream is empty, invalid, mixed, out of order, inconsistent with
        the context or incompatible with declared completion.

    Notes
    -----
    Host-only. Consumes the iterable and may read device arrays during capture,
    through the capture pair that picks each record's version from the
    context. It does not edit inputs, calculate metrics or publish files.
    """
    from marl_battlegrounds.evaluation._recording.capture import (
        _capture_evaluation_transition_unit,  # pyright: ignore[reportPrivateUsage]
        _capture_initial_evaluation_frame,  # pyright: ignore[reportPrivateUsage]
    )

    frames, transitions = _frames_from_packets(
        context,
        packets,
        capture_initial=_capture_initial_evaluation_frame,
        capture_transition=_capture_evaluation_transition_unit,
    )
    return build_replay_v5(
        context,
        cast(list[EvaluationFrameV4], frames),
        cast(list[EvaluationTransitionV2], transitions),
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
        completion_state=completion_state,
        end_or_failure_reason=end_or_failure_reason,
        failure_origin=failure_origin,
    )


__all__ = [
    "ReplayArtifactHeaderV5",
    "ReplayArtifactReferenceV5",
    "ReplayArtifactV5",
    "build_replay_v5",
    "replay_from_packets",
    "replay_reference_v5",
    "validate_replay_artifact_v5",
]
