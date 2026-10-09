"""Check six real native actor/full folders through accepted client turns.

Every fixture is initialized at step zero by the existing learner and save owners.
No rollout or optimizer update is performed. Actor exports name the actual full
checkpoint they came from. The targeted matrix covers saved scales, both spawn
frames, all/class/none sharing, both teams, exact public System action/memory
parity, fresh restart memory and saved replay identities. Separate source and
metadata checks use real native folders. These are CPU integration proofs, not
learning, playing-skill or GPU-speed claims.
"""

# pyright: reportPrivateUsage=false

import hashlib
import importlib.metadata
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.battle_client.test_battle_client_mixed_control import _host, _session
from tests.battle_client.test_battle_client_worker import _finish, _service
from tests.test_visual_debugger_systems import _send
from tests.training.test_training_checkpoints import _equal as _same_tree

from marl_battlegrounds._battle_client import methods
from marl_battlegrounds._battle_client.evaluation_bridge import (
    DebuggerActionSourceKindV1,
)
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.protocol import (
    FinishAndReviewCommandV1,
    KeyboardCommandV1,
    ResetCommandV1,
)
from marl_battlegrounds._battle_client.recording import (
    build_debugger_recording_specification_v1,
    recording_action_source,
)
from marl_battlegrounds._battle_client.replay_recorder import DebuggerReplayRecorder
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds._battle_client.system_worker import SystemWorker
from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.baselines.pqn import PQNActorVariables, PQNConfig
from marl_battlegrounds.baselines.qmix import QMIXActorVariables, QMIXConfig
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.environment import EnvironmentState
from marl_battlegrounds.evaluation.models import AssignedPolicySlotV2
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTree,
    System,
    SystemState,
    apply_systems,
    init_systems,
)
from marl_battlegrounds.evaluation.replay_io import (
    load_replay,
    preflight_replay_destination,
)
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.training import make_training_schedule, prepare_training_content
from marl_battlegrounds.training._compilation import execution_identity
from marl_battlegrounds.training._content import PreparedTrainingContent
from marl_battlegrounds.training._run_io import run_lock
from marl_battlegrounds.training.checkpoints import (
    artifact_identity,
    export_system,
    load_system,
    read_checkpoint_details,
    save_checkpoint,
)
from marl_battlegrounds.training.learner import init_learner
from marl_battlegrounds.training.pqn_learner import init_pqn_learner
from marl_battlegrounds.training.qmix_learner import init_qmix_learner

MATRIX = (
    ("mappo", "class", 0.01, "left"),
    ("ippo", "all", 1.0, "world"),
    ("ff_mappo", "all", 1.0, "left"),
    ("ff_ippo", "all", 0.01, "world"),
    ("qmix", "none", 0.01, "left"),
    ("pqn_vdn", "class", 1.0, "world"),
)


@pytest.fixture(scope="module")
def native_content() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module")
def native_case(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    native_content: PreparedTrainingContent,
) -> dict[str, Any]:
    method, sharing, scale, frame = request.param
    directory = tmp_path_factory.mktemp(f"native_{method}")
    run = directory / "run"
    run.mkdir()
    seed = 73 + [row[0] for row in MATRIX].index(method)
    common: dict[str, Any] = {
        "schedule": make_training_schedule(total_env_steps=20, num_envs=2),
        "seed": seed,
        "prepared": native_content,
        "metrics": "none",
        "recording": False,
        "keep_past": 0,
        "history_capture_capacity": 0,
    }
    settings: Any
    started = perf_counter()
    if method == "qmix":
        settings = QMIXConfig(
            rollout_length=2,
            buffer_size=4,
            min_buffer_size=2,
            sample_sequence_length=2,
            sample_batch_size=2,
            epochs=1,
            parameter_sharing=sharing,
            input_scale=scale,
            spawn_frame=frame,
        )
        collection, state = init_qmix_learner(**common, qmix=settings)
        options: dict[str, Any] = {"qmix": settings}
    elif method == "pqn_vdn":
        settings = PQNConfig(
            rollout_length=2,
            memory_window=1,
            epochs=1,
            num_minibatches=1,
            parameter_sharing=sharing,
            input_scale=scale,
            spawn_frame=frame,
        )
        collection, state = init_pqn_learner(**common, pqn=settings)
        options = {"pqn": settings}
    else:
        settings = PPOConfig(
            rollout_length=2,
            epochs=1,
            minibatches=1,
            parameter_sharing=sharing,
            input_scale=scale,
            spawn_frame=frame,
        )
        collection, state = init_learner(**common, ppo=settings, method=method)
        options = {"ppo": settings}
    jax.block_until_ready(state)
    initialization_seconds = perf_counter() - started
    metadata: dict[str, object] = {
        "run_id": f"untrained-client-{method}-{seed}",
        "attempt_id": "initialization-only",
        "parent_checkpoint": None,
        "config": {
            "method": method,
            "seed": seed,
            "num_envs": 2,
            "total_env_steps": 20,
            **{name: asdict(value) for name, value in options.items()},
        },
        "source": {
            "fixture_file": Path(__file__).name,
            "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "purpose": "Untrained client integration fixture",
        },
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("jax", "flax", "optax")
        },
        "execution": execution_identity(),
    }
    with run_lock(run):
        full = save_checkpoint(
            run, collection, state, metadata=metadata, method=method, **options
        )
    details = read_checkpoint_details(full)
    assert all(count == 0 for count in details["counters"].values())
    actor_variables = load_system(full).variables
    if method == "qmix":
        assert isinstance(actor_variables, QMIXActorVariables)
        actor_variables = actor_variables.params
    elif method == "pqn_vdn":
        assert isinstance(actor_variables, PQNActorVariables)
        actor_variables = actor_variables.network
    assert any(
        np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(actor_variables)
    )
    actor = export_system(
        actor_variables,
        directory / "actor",
        method=method,
        input_scale=scale,
        spawn_frame=frame,
        parameter_sharing=sharing,
        metadata={
            "run_id": metadata["run_id"],
            "seed": seed,
            "env_steps": 0,
            "optimizer_steps": 0,
            "checkpoint_id": details["checkpoint_id"],
            "purpose": "Untrained client integration fixture",
        },
    )
    report = {
        "method": method,
        "parameter_sharing": sharing,
        "input_scale": scale,
        "spawn_frame": frame,
        "seed": seed,
        "directory": str(directory),
        "actor": str(actor),
        "full": str(full),
        "trained_env_steps": 0,
        "optimizer_steps": 0,
        "initialization_seconds": initialization_seconds,
        "fixture_seconds": perf_counter() - started,
        "actor_identity": artifact_identity(actor),
        "full_identity": artifact_identity(full),
    }
    (directory / "fixture.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _fresh_memory(session: DebuggerSession) -> SystemState:
    wrapped = session.environment_state
    assert wrapped is not None and session.systems is not None
    seed = session.evaluation_context.seed_protocol.focal_policy_seed
    assert seed is not None
    return init_systems(
        *session.systems,
        Observations(wrapped.observation, wrapped.source_availability),
        wrapped,
        jax.random.key(seed),
    )


def _expected(
    session: DebuggerSession, *, memory: SystemState | None = None
) -> tuple[Action, SystemState, tuple[PolicyTree, PolicyTree]]:
    wrapped = session.environment_state
    assert wrapped is not None and session.systems is not None
    chosen = session.system_memory if memory is None else memory
    assert chosen is not None
    seed = session.evaluation_context.seed_protocol.focal_policy_seed
    assert seed is not None
    return apply_systems(
        *session.systems,
        chosen,
        Observations(wrapped.observation, wrapped.source_availability),
        wrapped,
        jax.random.fold_in(
            jax.random.key(seed),
            session.current_evaluation_frame.frame_index,
        ),
    )


@pytest.mark.parametrize(
    "native_case", MATRIX, indirect=True, ids=[row[0] for row in MATRIX]
)
def test_native_actor_and_full_routes_accept_exact_actions_and_restart_fresh(
    native_case: dict[str, Any],
) -> None:
    directory = Path(native_case["directory"])
    started = perf_counter()
    session = _session(
        prepare_battle_setup(
            map_id=0,
            max_steps=6,
            seed=193,
            team_a=Path(native_case["actor"]),
            team_b=Path(native_case["full"]),
        ),
        capture=True,
    )
    setup_seconds = perf_counter() - started
    assert session.systems is not None
    pair = session.systems
    assert pair[0] is not pair[1]
    assert pair[0].apply is pair[1].apply
    _same_tree(pair[0].variables, pair[1].variables)
    for index, route in enumerate(("actor", "full")):
        identity = native_case[f"{route}_identity"]
        assert pair[index].checkpoint == identity["actor_digest"]
        assert identity["input_scale"] == native_case["input_scale"]
        assert identity["spawn_frame"] == native_case["spawn_frame"]
    _same_tree(session.system_memory, _fresh_memory(session))
    recorder = DebuggerReplayRecorder(
        specification=build_debugger_recording_specification_v1(
            action_source_kind=cast(
                DebuggerActionSourceKindV1,
                recording_action_source(session.evaluation_context),
            ),
            runtime_provenance=capture_runtime_provenance(
                session.evaluation_context.code_revision.package_version,
                policy_execution_included=True,
            ),
        ),
        destination=preflight_replay_destination(
            directory / "accepted.marlbg-replay.json"
        ),
        context=session.evaluation_context,
        initial_frame=session.current_evaluation_frame,
    )
    worker = SystemWorker(caller_thread=True)
    worker.install(session, directory=directory / "client")
    service = DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        recorder=recorder,
        system_worker=worker,
    )
    turns = []
    recurrent = not native_case["method"].startswith("ff_")
    try:
        for turn in range(2):
            before = service.session
            expected_started = perf_counter()
            expected_action, expected_memory, _ = _expected(before)
            jax.block_until_ready((expected_action, expected_memory))
            expected_seconds = perf_counter() - expected_started
            if turn == 1 and recurrent:
                _, reset_each_turn, _ = _expected(before, memory=_fresh_memory(before))
                assert any(
                    not np.array_equal(left, right)
                    for left, right in zip(
                        jax.tree.leaves(expected_memory.team_a),
                        jax.tree.leaves(reset_each_turn.team_a),
                        strict=True,
                    )
                )
            turn_started = perf_counter()
            _send(service, KeyboardCommandV1(key="Enter"))
            _finish(service, worker)
            assert service.session.current_evaluation_frame.frame_index == turn + 1
            _same_tree(service.session.system_memory, expected_memory)
            incoming = service.session.incoming_evaluation_view
            assert incoming is not None
            transition = incoming.transition
            facts = transition.facts.action_acceptance_facts
            for name in ("move", "select_target", "use_ultimate"):
                wanted = tuple(np.asarray(getattr(expected_action, name)).tolist())
                assert getattr(facts.submitted_joint_action, name) == wanted
                assert getattr(facts.accepted_joint_action, name) == wanted
            assert not any(facts.submitted_action_tuple_is_out_of_domain_by_actor)
            turns.append(
                {
                    "frame_index": turn + 1,
                    "reference_seconds": expected_seconds,
                    "client_turn_seconds": perf_counter() - turn_started,
                    "accepted": facts.accepted_joint_action.model_dump(mode="json"),
                }
            )
        carried = service.session.system_memory
        assert carried is not None
        if recurrent:
            assert any(
                np.any(np.asarray(leaf) != 0)
                for leaf in jax.tree.leaves(carried.team_a)
            )
        else:
            assert carried.team_a == carried.team_b == ()
        save_started = perf_counter()
        _send(service, FinishAndReviewCommandV1())
        assert recorder.lifecycle == "saved"
        save_seconds = perf_counter() - save_started
        saved = load_replay(directory / "accepted.marlbg-replay.json").replay
        assert len(saved.transitions) == 2
        for slot, assignment in enumerate(saved.header.context.policy_assignments):
            assert isinstance(assignment, AssignedPolicySlotV2)
            assert assignment.policy_kind == "system"
            assert assignment.checkpoint_digest == pair[slot // 5].checkpoint
        _send(service, ResetCommandV1())
        _finish(service, worker)
        restarted = service.session
        assert restarted.systems is pair
        assert restarted.system_memory is not carried
        assert restarted.current_evaluation_frame.frame_index == 0
        _same_tree(restarted.system_memory, _fresh_memory(restarted))
        expected_action, expected_memory, _ = _expected(restarted)
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service, worker)
        _same_tree(service.session.system_memory, expected_memory)
        incoming = service.session.incoming_evaluation_view
        assert incoming is not None
        for name in ("move", "select_target", "use_ultimate"):
            assert getattr(
                incoming.transition.facts.action_acceptance_facts.accepted_joint_action,
                name,
            ) == tuple(np.asarray(getattr(expected_action, name)).tolist())
        report = {
            "method": native_case["method"],
            "routes": ["actor", "full"],
            "setup_seconds": setup_seconds,
            "turns": turns,
            "save_seconds": save_seconds,
            "recurrent_memory_advances": recurrent,
            "restart_fresh": True,
            "same_apply_hook": True,
            "identities": [method.checkpoint for method in pair],
            "assignments": [
                item.model_dump(mode="json")
                for item in saved.header.context.policy_assignments
            ],
        }
        (directory / "client_proof.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    finally:
        service.close()


@pytest.mark.parametrize("native_case", [MATRIX[0]], indirect=True, ids=["mappo"])
def test_real_native_repeated_sources_and_distinct_objects(
    native_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = Path(native_case["actor"])
    setup = prepare_battle_setup(
        map_id=0, max_steps=3, team_a=actor, team_b=(str(actor),) * 5
    )
    original = methods.load_method
    calls: list[str] = []

    def count_load(source: str) -> System | Policy:
        calls.append(source)
        return original(source)

    monkeypatch.setattr(methods, "load_method", count_load)
    resolved = methods.resolve_controllers(setup)
    assert calls == [str(actor)]
    assert len(resolved.methods) == 1
    assert resolved.systems is not None and resolved.systems[0] is resolved.systems[1]
    first, second = load_system(actor), load_system(actor)
    assert first is not second and first.checkpoint == second.checkpoint
    direct = methods.resolve_controllers(replace(setup, team_a=first, team_b=second))
    assert direct.methods == (first, second)
    assert direct.methods[0] is first and direct.methods[1] is second
    assert calls == [str(actor)]


@pytest.mark.parametrize("native_case", [MATRIX[0]], indirect=True, ids=["mappo"])
def test_native_metadata_refused_before_client_session(
    native_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    actor = Path(native_case["actor"])
    malformed = tmp_path / "malformed"
    shutil.copytree(actor, malformed)
    description = malformed / "actor_details.json"
    description.write_text("{not valid JSON\n")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        methods.resolve_controllers(prepare_battle_setup(map_id=0, team_a=malformed))
    incompatible = tmp_path / "incompatible"
    shutil.copytree(actor, incompatible)
    description = incompatible / "actor_details.json"
    value = json.loads(description.read_text())
    value["schemas"]["model"] = "not-a-native-model"
    description.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        methods.resolve_controllers(prepare_battle_setup(map_id=0, team_a=incompatible))


@pytest.mark.parametrize("native_case", [MATRIX[0]], indirect=True, ids=["mappo"])
def test_native_host_and_manual_accept_one_mixed_turn(
    native_case: dict[str, Any],
) -> None:
    def array(value: PolicyTree) -> jax.Array:
        return jnp.asarray(value)

    records: list[dict[str, Any]] = []
    native = load_system(native_case["actor"])
    host = _host(records)
    session = _session(
        prepare_battle_setup(
            map_id=0,
            max_steps=3,
            team_a=("manual", native, native, native, host),
            team_b=Path(native_case["full"]),
        )
    )
    expected_action, expected_memory, _ = _expected(session)
    service, worker = _service(session)
    try:
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service, worker)
        actual_memory = service.session.system_memory
        assert actual_memory is not None
        # The client removes Manual from the policy trace after composition.
        # Compare carried method memory; existing trace tests cover that masking.
        for actual, expected in (
            (actual_memory.team_a, expected_memory.team_a),
            (actual_memory.team_b, expected_memory.team_b),
        ):
            _same_tree(jax.tree.map(array, actual), jax.tree.map(array, expected))
        incoming = service.session.incoming_evaluation_view
        assert incoming is not None
        for name in ("move", "select_target", "use_ultimate"):
            assert getattr(
                incoming.transition.facts.action_acceptance_facts.accepted_joint_action,
                name,
            ) == tuple(np.asarray(getattr(expected_action, name)).tolist())
        assert session.manual_slots == (0,)
        assert len([row for row in records if row["phase"] == "apply"]) == 2
        service.current_frame()
        service.current_frame()
        assert len([row for row in records if row["phase"] == "apply"]) == 2
    finally:
        service.close()


@pytest.mark.parametrize("native_case", [MATRIX[0]], indirect=True, ids=["mappo"])
def test_native_public_jit_reuses_structure_with_changed_values(
    native_case: dict[str, Any],
) -> None:
    session = _session(
        prepare_battle_setup(
            map_id=0,
            max_steps=3,
            team_a=Path(native_case["actor"]),
            team_b=Path(native_case["full"]),
        )
    )
    pair, wrapped, memory = (
        session.systems,
        session.environment_state,
        session.system_memory,
    )
    assert pair is not None and wrapped is not None and memory is not None
    traces: list[str] = []

    def apply(
        a: PolicyTree,
        b: PolicyTree,
        carried: SystemState,
        state: EnvironmentState,
        key: jax.Array,
    ) -> tuple[Action, SystemState, tuple[PolicyTree, PolicyTree]]:
        traces.append("trace")
        return apply_systems(
            *pair,
            carried,
            Observations(state.observation, state.source_availability),
            state,
            key,
            variables_a=a,
            variables_b=b,
        )

    compiled = cast(
        Callable[..., tuple[Action, SystemState, tuple[PolicyTree, PolicyTree]]],
        jax.jit(apply),
    )

    def shift(leaf: jax.Array, *, amount: float) -> jax.Array:
        return leaf + jnp.float32(amount)

    times: list[float] = []
    current = memory
    for index in range(3):
        variables = jax.tree.map(
            partial(shift, amount=index * 0.001), pair[0].variables
        )
        started = perf_counter()
        result = compiled(
            variables, pair[1].variables, current, wrapped, jax.random.key(500 + index)
        )
        jax.block_until_ready(result)
        times.append(perf_counter() - started)
        current = result[1]
    assert traces == ["trace"]
    (Path(native_case["directory"]) / "public_jit_reuse.json").write_text(
        json.dumps(
            {
                "method": "mappo",
                "parameter_sharing": "class",
                "calls": 3,
                "python_traces": 1,
                "seconds": times,
                "changed_values": ["actor weights", "memory", "random keys"],
                "scope": (
                    "Public apply_systems under one caller-owned jit; "
                    "not a live-client throughput claim"
                ),
            },
            indent=2,
        )
        + "\n"
    )
