"""Check Capture the Flag recording contexts and replay V5.

Contracts: build_recording_context records a Capture the Flag config as context V5, with
the task named "ctf" with its capture threshold and the reward contract "core-ctf"; a
registered CTF map keeps its map identity, and a custom one gets a replay file name
starting "ctf_custom_map_". A captured Capture the Flag game seals into replay V5 (frame
V4, transition V2), saves as its exact canonical bytes and loads back equal, with
reference version 5; PreparedReplay accepts it. build_replay_v5 refuses a Team
Deathmatch context, build_replay_v4 refuses a Capture the Flag transition, and scalar
replay analysis refuses replay V5. The shared transition view joins a Capture the Flag
transition, and an actor's POV V4 export (POV V3 frames, an axis mapping naming context
5, frame 4 and transition 2) validates against its source replay and has canonical
bytes. Real full games are checked once Core scores Capture the Flag (his C1.3 and
C1.4); these games stop before the horizon. Host only, on CPU.
"""

from pathlib import Path

import jax
import pytest
from tests.helpers.evaluation_fixtures import (
    current_captured_evaluation_trajectory,
    neutral_action,
)

from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.evaluation._metrics.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation._recording.capture import (
    capture_evaluation_transition_unit,
    capture_initial_evaluation_frame,
)
from marl_battlegrounds.evaluation._recording.map_identity import recorded_map
from marl_battlegrounds.evaluation._recording.recording_context import (
    build_recording_context,
)
from marl_battlegrounds.evaluation._replays.pov import (
    ActorPovFrameV3,
    ActorPovReplayArtifactV4,
    canonical_actor_pov_replay_json_bytes_v4,
    export_actor_pov_replay_v4,
    validate_actor_pov_replay_against_replay_v4,
)
from marl_battlegrounds.evaluation._replays.replay_v4 import build_replay_v4
from marl_battlegrounds.evaluation._replays.replay_v5 import (
    ReplayArtifactV5,
    build_replay_v5,
    replay_reference_v5,
)
from marl_battlegrounds.evaluation.analysis import analyze_replay
from marl_battlegrounds.evaluation.models import (
    CodeRevisionV2,
    EvaluationEpisodeContextV5,
    EvaluationFrameV4,
    EvaluationTransition,
    EvaluationTransitionV2,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.replay_io import (
    LoadedReplay,
    PreparedReplay,
    generated_replay_filename,
    load_replay,
    save_replay,
)
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.tasks import (
    AgentClassName,
    make_standard_capture_the_flag_config,
)

_A: tuple[AgentClassName, ...] = ("mage", "warrior")
_B: tuple[AgentClassName, ...] = ("rogue",)
_CONFIG = make_standard_capture_the_flag_config(
    map_id=42, team_a_roster=_A, team_b_roster=_B, max_steps=50
)
_RUNTIME = capture_runtime_provenance("0.0.0")


def _context(**episode: object) -> EvaluationEpisodeContextV5:
    context, _ = build_recording_context(
        jax.device_get(_CONFIG),
        run_id="run-1",
        phase="evaluation",
        pass_id="1",
        episode={"episode_id": 7, **episode},
        policies={"team_a": "Alpha", "team_b": "Beta"},
        details={
            "code_revision": CodeRevisionV2(package_version="0.0.0").model_dump(
                mode="json"
            ),
            "runtime_provenance": _RUNTIME.model_dump(mode="json"),
            "execution_information_mode": "no_shared_obs",
        },
    )
    assert type(context) is EvaluationEpisodeContextV5
    return context


def test_a_capture_the_flag_recording_context_names_its_task() -> None:
    context = _context(map_id=42)
    task = context.identity.task
    assert (task.identifier, context.canonical_reward_mode.identifier) == (
        "ctf",
        "core-ctf",
    )
    assert context.resolved_env_config.capture_the_flag_score_threshold == 3
    assert recorded_map(context).map_id == 42
    custom = _context()
    assert recorded_map(custom).display_name == "Custom Map"
    filename = generated_replay_filename(custom, "f" * 64, episode_id=7)
    assert filename.startswith("ctf_custom_map_")


@pytest.fixture(scope="module")
def replay() -> ReplayArtifactV5:
    context = _context(map_id=42)
    state, observation, mask, _ = reset(_CONFIG, jax.random.PRNGKey(0))
    frames = [capture_initial_evaluation_frame(context, state, observation, mask)]
    transitions: list[EvaluationTransition] = []
    for index in range(3):
        state, observation, reward, done, mask, info = step(
            _CONFIG, state, mask, neutral_action(), jax.random.PRNGKey(index + 1)
        )
        transition, frame = capture_evaluation_transition_unit(
            context,
            frames[-1],
            state,
            observation,
            mask,
            info.transition_facts,
            reward,
            done,
        )
        frames.append(frame)
        transitions.append(transition)
    ctf_frames = [frame for frame in frames if isinstance(frame, EvaluationFrameV4)]
    ctf_rows = [row for row in transitions if isinstance(row, EvaluationTransitionV2)]
    assert (len(ctf_frames), len(ctf_rows)) == (len(frames), len(transitions))
    return build_replay_v5(context, ctf_frames, ctf_rows, runtime_provenance=_RUNTIME)


def test_replay_v5_saves_and_loads_its_exact_bytes(
    tmp_path: Path, replay: ReplayArtifactV5
) -> None:
    assert replay.completion.completion_state == "partial"
    path = tmp_path / "ctf.marlbg-replay.json"
    saved = save_replay(replay, path)
    loaded = load_replay(path)
    assert type(loaded.replay) is ReplayArtifactV5
    assert loaded.replay == replay
    assert path.read_bytes() == canonical_json_bytes(replay)
    assert PreparedReplay(replay).replay_json_bytes == path.read_bytes()
    reference = replay_reference_v5(replay)
    assert (reference.schema_version, reference.replay_schema_version) == (5, 5)
    assert reference.canonical_byte_length == saved.replay_byte_length


def test_replay_v5_refuses_other_contexts_and_tdm_analysis(
    replay: ReplayArtifactV5,
) -> None:
    tdm = current_captured_evaluation_trajectory()
    with pytest.raises(TypeError, match="exact context V5"):
        build_replay_v5(
            tdm.context,  # pyright: ignore[reportArgumentType]
            tdm.frames,  # pyright: ignore[reportArgumentType]
            tdm.transitions,  # pyright: ignore[reportArgumentType]
            runtime_provenance=_RUNTIME,
        )
    with pytest.raises(ValueError, match="supports TDM only"):
        analyze_replay(LoadedReplay(replay))


def test_a_player_view_exports_from_replay_v5(replay: ReplayArtifactV5) -> None:
    view = EvaluationTransitionViewV1(
        context=replay.header.context,
        start_frame=replay.frames[0],
        transition=replay.transitions[0],
        successor_frame=replay.frames[1],
    )
    assert type(view.transition) is EvaluationTransitionV2
    artifact = export_actor_pov_replay_v4(replay, global_slot=0)
    assert type(artifact) is ActorPovReplayArtifactV4
    mapping = artifact.content.axis_mapping
    assert (
        mapping.source_context_schema_version,
        mapping.source_frame_schema_version,
        mapping.source_transition_schema_version,
    ) == (5, 4, 2)
    assert {type(frame) for frame in artifact.content.frames} == {ActorPovFrameV3}
    validate_actor_pov_replay_against_replay_v4(artifact, replay)
    assert canonical_actor_pov_replay_json_bytes_v4(artifact) == canonical_json_bytes(
        artifact
    )
    with pytest.raises(TypeError, match="exact transition V1"):
        build_replay_v4(
            current_captured_evaluation_trajectory().context,
            [],
            list(replay.transitions),
            runtime_provenance=_RUNTIME,
        )
