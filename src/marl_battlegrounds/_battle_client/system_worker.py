"""Own one bounded live worker and declared System installations.

The service submits setup or one joint decision, then explicitly accepts or
abandons its result. Each installation keeps its own context and resource stack.
Factory work uses its owned context. Borrowed work runs in the original caller
context through a small queue. Only that execution owner enters/exits scopes.
Model-call files stay beside the
recorder's destination; they never decide whether replay data is safely saved.
No worker, factory or connection starts until the service requests work.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack, nullcontext
from contextvars import Context
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, get_ident
from typing import Any, Literal, TypeVar
from uuid import uuid4

import jax

from marl_battlegrounds._battle_client.input import InputDispatchResult
from marl_battlegrounds._battle_client.methods import LoadedControllers
from marl_battlegrounds._battle_client.model import (
    DebuggerSession,
    is_system_controller,
)
from marl_battlegrounds._method_loading import load_factory
from marl_battlegrounds.evaluation.host_evidence import HostRun
from marl_battlegrounds.evaluation.policy_execution import Policy, System, shared_policy

_Result = TypeVar("_Result")


class AbandonedSystemDecisionError(RuntimeError):
    """The service did not adopt this prepared joint decision."""


@dataclass(eq=False)
class SystemInstallation:
    """Keep one owned setup's resources and evidence on its worker context.

    loaded contains actual factory returns for alias and singleton checks.
    session is the prepared initial snapshot. cleanup and decision must only be
    used through run on the single execution owner. None context means the exact
    original caller context, never a copied one. closed becomes true after cleanup
    succeeds. resources holds method-lifetime leases separately from match evidence
    in cleanup, so an adopted restart can retain its exact methods and context.
    A failed cleanup remains owned and must not be hidden.
    """

    context: Context | None = field(default_factory=Context)
    cleanup: ExitStack = field(default_factory=ExitStack)
    resources: ExitStack = field(default_factory=ExitStack)
    loaded: dict[str, System | Policy] = field(
        default_factory=lambda: dict[str, System | Policy]()
    )
    session: DebuggerSession | None = None
    evidence: HostRun | None = None
    decision: AbstractContextManager[object] | None = None
    directory: Path | None = None
    metadata: dict[str, object] = field(default_factory=lambda: dict[str, object]())
    closed: bool = False
    close_error: BaseException | None = None
    cancellation: Event = field(default_factory=Event)
    saved_prefix_requests: set[tuple[str, str, int]] = field(
        default_factory=lambda: set[tuple[str, str, int]]()
    )
    """Save acknowledgements already queued for this episode, including pending ones."""

    def run(self, action: Callable[..., _Result], *args: object) -> _Result:
        """Use the owned context, or the original caller context for borrowed work.

        Caller installations bind evidence for this callback only. Their resource
        leases and decision scopes are entered and exited by the caller thread.
        No copied Context is entered for a borrowed method.
        """

        def bound() -> _Result:
            """Bind only this callback's evidence on its existing execution owner."""
            binding = nullcontext() if self.evidence is None else self.evidence.bind()
            with binding:
                return action(*args)

        return self.context.run(bound) if self.context is not None else bound()

    def check_cancelled(self) -> None:
        """Avoid starting another callback after the service has cancelled setup."""
        if self.cancellation.is_set():
            raise AbandonedSystemDecisionError("System setup was cancelled")

    def close(self) -> None:
        """Close this installation on its context after any current call settles."""
        if self.closed:
            return
        if self.close_error is not None:
            raise RuntimeError(
                "System cleanup failed; ownership is still retained"
            ) from self.close_error
        errors: list[BaseException] = []
        for action in (
            lambda: self.settle(False),
            self.cleanup.close,
            self.resources.close,
        ):
            try:
                action()
            except BaseException as error:
                errors.append(error)
        if errors:
            self.close_error = BaseExceptionGroup("System cleanup failed", errors)
            raise self.close_error
        self.closed = True

    def settle(self, played: bool) -> None:
        """Finish the open decision once, then flush its separate call evidence.

        played comes only from service adoption, including a later recording
        error after an accepted step. False abandons the answer and its history.
        This does not mark a game or replay as saved.
        """
        scope, self.decision = self.decision, None
        if scope is not None:
            if played:
                scope.__exit__(None, None, None)
            else:
                error = AbandonedSystemDecisionError(
                    "Prepared decision was not accepted"
                )
                scope.__exit__(type(error), error, error.__traceback__)
        if self.evidence is not None:
            self.evidence.flush()

    def write_policy_trace(self, session: DebuggerSession) -> None:
        """Append this accepted decision's existing trace only when recording is on.

        The worker calls once after real adoption, outside HTTP/service locks.
        Keep the numerical PolicyTrace fields and the accepted frame index in one
        JSON line. Setup descriptors in session.json resolve component IDs; -1
        stays unreported, including rows supplied by Manual. This is accepted
        live evidence, not a claim that the replay prefix has been saved.
        Disabled recording performs no transfer or file work. Write failures
        propagate through the existing failed-operation fence without undoing
        the accepted game. The caller never retries a partially written row.
        """
        if self.directory is None or session.system_memory is None:
            return
        trace = jax.device_get(session.system_memory.policy_trace)
        fields = {name: getattr(trace, name).tolist() for name in trace._fields}
        row = {
            "accepted_frame_index": session.current_evaluation_frame.frame_index,
            **fields,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "policy_trace.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.metadata["last_policy_trace"] = fields

    def write_details(self) -> None:
        """Atomically save setup and accepted evidence, only when recording is on."""
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / "session.json"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(self.metadata, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)


@dataclass(eq=False)
class PendingSystemOperation:
    """One setup or decision, retained until adoption or cancellation cleanup.

    base is the exact accepted snapshot used to start work. future finishes when
    a result is ready, not when it is played. cleanup_future keeps a cancelled
    slot occupied until late work and resource cleanup both finish. reuses_methods
    marks unchanged-method restart and transfers leases only after adoption.
    prepared_replacement marks a trusted new setup whose exact prepared seed and
    source may replace the previous match; ordinary reset keeps the old source.
    """

    kind: Literal["setup", "decision"]
    base: DebuggerSession
    installation: SystemInstallation
    future: Future[SystemInstallation] | Future[InputDispatchResult]
    identifier: str = field(default_factory=lambda: uuid4().hex)
    cancelled: bool = False
    adopted: bool = False
    cleanup_future: Future[None] | None = None
    acceptance_future: Future[None] | None = None
    replaced: SystemInstallation | None = None
    on_installed: Callable[[], None] | None = None
    reuses_methods: bool = False
    prepared_replacement: bool = False
    replacement_recording_identity: object | None = None
    """Exact recorder protected by setup approval; None means capture was disabled."""


class SystemWorker:
    """Run one setup or decision on its factory worker or original caller thread.

    Constructing this helper starts nothing. The service may hold one active
    installation and one candidate. It must settle the current operation before
    submitting another. Cancellation cannot forcibly stop arbitrary Python code;
    its slot remains occupied until that code returns and cleanup finishes.
    """

    def __init__(self, *, caller_thread: bool = False) -> None:
        """Start an idle executor worker or an original-caller queue.

        caller_thread=False preserves private factory execution. True never
        starts an executor: drain must run on this creating thread and in its
        original Python context. Construction invokes no method or resource.
        """
        self.caller_thread = caller_thread
        self._owner_thread = get_ident()
        self._queue: deque[tuple[Future[Any], Callable[[], Any]]] = deque()
        self._queue_lock = Lock()
        self.failed = False
        self._executor: ThreadPoolExecutor | None = None
        self.operation: PendingSystemOperation | None = None
        self.active: SystemInstallation | None = None
        self._closed = False
        self.shutdown_futures: list[Future[None]] = []
        self.recording_futures: list[Future[None]] = []

    def _submit(
        self, installation: SystemInstallation, action: Callable[[], _Result]
    ) -> Future[_Result]:
        """Schedule one callback without copying the borrowed caller context."""
        if not self.caller_thread:
            return self._pool().submit(installation.run, action)
        if self._closed:
            raise RuntimeError("System worker is closed")
        future: Future[_Result] = Future()
        with self._queue_lock:
            self._queue.append((future, lambda: installation.run(action)))
        return future

    def drain(self) -> bool:
        """Run one queued callback on the original caller thread, outside locks.

        Return False when no callback remains. Ordinary callback failures stay in
        their Futures for service adoption. Ctrl-C and other process interrupts
        return to the caller. After close, an empty queue reports retained cleanup
        failures. This method never enters a copied Python context.
        """
        if not self.caller_thread:
            return False
        if get_ident() != self._owner_thread:
            raise RuntimeError("Drain System work on the original caller thread")
        with self._queue_lock:
            item = self._queue.popleft() if self._queue else None
        if item is None:
            if self._closed:
                futures = [*self.shutdown_futures, *self.recording_futures]
                if self.operation is not None:
                    futures.extend(
                        future
                        for future in (
                            self.operation.cleanup_future,
                            self.operation.acceptance_future,
                        )
                        if future is not None
                    )
                errors = [
                    error
                    for future in futures
                    if future.done() and (error := future.exception()) is not None
                ]
                if errors:
                    raise BaseExceptionGroup("System cleanup failed", errors)
            return False
        future, action = item
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(action())
            except BaseException as error:
                future.set_exception(error)
                if not isinstance(error, Exception):
                    self.failed = True
                    raise
        self._prune_successes()
        return True

    def _prune_successes(self) -> None:
        """Drop completed successful bookkeeping Futures; retain all failures."""
        with self._queue_lock:
            for futures in (self.shutdown_futures, self.recording_futures):
                futures[:] = [
                    future
                    for future in futures
                    if not future.done() or future.exception() is not None
                ]

    def _describe(
        self, installation: SystemInstallation, session: DebuggerSession
    ) -> None:
        """Attach one match's existing IDs and evidence without closing methods."""
        installation.session = session
        identity = session.evaluation_context.identity
        installation.metadata.update(
            schema_version=1,
            run_id=identity.run_id,
            episode_id=identity.episode_id,
            local_episode_id=session.numeric_episode_id,
            accepted_turns=0,
            saved_prefixes=[],
            systems={
                str(team): registration
                for team, (_, registration) in enumerate(
                    session.system_registrations or ()
                )
            },
            system_ids=session.system_ids,
            attempts_are_not_saved_replays=True,
        )
        if installation.directory is not None and session.system_memory is not None:
            installation.metadata["policy_trace_file"] = "policy_trace.jsonl"
        installation.evidence = HostRun(
            identity.run_id,
            "devclient",
            identity.episode_id,
            installation.directory,
            installation.metadata,
        )
        installation.cleanup.callback(installation.evidence.close)

    @staticmethod
    def _enter_resources(
        installation: SystemInstallation, cleanup: ExitStack, session: DebuggerSession
    ) -> None:
        """Add match recording scopes around existing method-lifetime leases.

        Owned work already holds False scopes in this exact Context, including
        during restart before lease transfer. Without capture, no extra scope is
        needed. Borrowed caller behavior retains its existing nested leases.
        """
        if installation.context is not None and installation.directory is None:
            return
        seen: set[int] = set()
        for method in session.systems or ():
            if id(method) not in seen and method.resource_scope is not None:
                cleanup.enter_context(
                    method.resource_scope(installation.directory is not None)
                )
            seen.add(id(method))

    def install(self, session: DebuggerSession, *, directory: Path | None) -> None:
        """Adopt one initial borrowed-System or owned legacy session.

        Caller-thread installs require borrowed methods and keep their original
        thread/context. Their launcher already holds method lifetime leases.
        Owned installs accept only a legacy session without Systems; the private
        launcher uses this before serving so Manual/Random share the same pump.
        directory is the recorder's evidence folder or None when capture is off.
        This initial setup may wait for the worker; never call it under HTTP locks.
        Errors close the candidate; supplied method objects are never closed here.
        """
        if get_ident() != self._owner_thread:
            raise RuntimeError("Install initial methods on the launch thread")
        self._idle()
        if self._closed or self.failed or self.active is not None:
            raise RuntimeError("Install one live session on an unused worker")
        if self.caller_thread:
            if session.systems is None or not session.borrow_systems:
                raise ValueError(
                    "Caller-thread installations must retain supplied methods"
                )
        elif session.systems is not None:
            raise ValueError(
                "Owned initial installation accepts legacy controllers only"
            )
        candidate = SystemInstallation(
            context=None if self.caller_thread else Context(), directory=directory
        )

        def initialize() -> None:
            """Bind the initial evidence on its exact execution owner."""
            try:
                self._describe(candidate, session)
                assert candidate.evidence is not None
                with candidate.evidence.bind(), ExitStack() as cleanup:
                    self._enter_resources(candidate, cleanup, session)
                candidate.write_details()
            except BaseException:
                self.failed = True
                candidate.close()
                raise

        if self.caller_thread:
            initialize()
        else:
            try:
                self._submit(candidate, initialize).result()
            except BaseException:
                self.close()
                raise
        self.active = candidate

    def restart(
        self,
        base: DebuggerSession,
        build: Callable[[], DebuggerSession],
        *,
        directory: Path | None,
    ) -> PendingSystemOperation:
        """Queue fresh memory using the installed methods and their original context.

        No method is loaded, copied or closed. Owned method leases transfer only
        after adoption; caller-owned outer leases stay with the launcher. The old
        accepted session remains inspectable. Failed initialization fences reuse.
        """
        self._idle()
        if self.failed or self.active is None:
            raise RuntimeError("Load fresh methods after failed System work")
        candidate = SystemInstallation(context=self.active.context, directory=directory)

        def initialize() -> SystemInstallation:
            """Build one replacement in the caller's unchanged Python context."""
            try:
                candidate.check_cancelled()
                session = build()
                candidate.check_cancelled()
                if session.systems is not base.systems and any(
                    new is not old
                    for new, old in zip(
                        session.systems or (), base.systems or (), strict=True
                    )
                ):
                    raise ValueError("Restart must retain the installed methods")
                self._describe(candidate, session)
                assert candidate.evidence is not None
                with candidate.evidence.bind(), ExitStack() as cleanup:
                    self._enter_resources(candidate, cleanup, session)
                candidate.write_details()
                return candidate
            except BaseException as error:
                self.failed = True
                try:
                    candidate.close()
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "Restart failed and its match cleanup also failed",
                        [error, cleanup_error],
                    ) from None
                raise

        operation = PendingSystemOperation(
            "setup",
            base,
            candidate,
            self._submit(candidate, initialize),
            reuses_methods=True,
        )
        self.operation = operation
        return operation

    @staticmethod
    def _report_failure(future: Future[None]) -> None:
        """Log asynchronous recording/cleanup failures even after HTTP shutdown."""
        error = future.exception()
        if error is not None:
            logging.getLogger(__name__).error(
                "DevClient System cleanup or recording failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _pool(self) -> ThreadPoolExecutor:
        """Create the single worker on first selected use; reject use after close."""
        if self._closed:
            raise RuntimeError("System worker is closed")
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="marl-devclient"
            )
        return self._executor

    def _idle(self) -> None:
        """Reject another setup or decision while the bounded slot is occupied."""
        if self.operation is not None:
            raise RuntimeError("A System operation is still pending")

    def _prepare_owned(
        self,
        base: DebuggerSession,
        build: Callable[[SystemInstallation], DebuggerSession],
        *,
        directory: Path | None,
    ) -> PendingSystemOperation:
        """Queue one private replacement in a separate owned context.

        build loads validated methods, enters their lifetime leases on the supplied
        installation and returns its initialized session. It runs outside HTTP
        locks. The accepted installation remains owned until explicit adoption.
        """
        self._idle()
        if self.failed:
            raise RuntimeError("Start a fresh client launch after failed method work")
        if self.caller_thread:
            raise RuntimeError("Borrowed launch assignments cannot load replacements")
        candidate = SystemInstallation(directory=directory)

        def initialize() -> SystemInstallation:
            """Create match evidence while retaining method lifetime leases."""
            try:
                candidate.check_cancelled()
                session = build(candidate)
                candidate.check_cancelled()
                self._describe(candidate, session)
                assert candidate.evidence is not None
                with candidate.evidence.bind(), ExitStack() as cleanup:
                    self._enter_resources(candidate, cleanup, session)
                candidate.write_details()
                return candidate
            except BaseException:
                candidate.close()
                raise

        operation = PendingSystemOperation(
            "setup", base, candidate, self._submit(candidate, initialize)
        )
        self.operation = operation
        return operation

    def prepare(
        self,
        base: DebuggerSession,
        declarations: Mapping[str, str],
        controllers: tuple[str, str],
        build: Callable[[tuple[System | None, System | None]], DebuggerSession],
        *,
        directory: Path | None,
    ) -> PendingSystemOperation:
        """Load selected launcher-declared factories on the existing private worker.

        declarations maps bounded aliases to trusted Python references; browser
        text does not add declarations. controllers follows Team A/B order and
        build initializes their prepared pair. directory is the recorder-owned
        evidence destination, or None. Each replacement must load fresh objects;
        unchanged-method restart uses restart instead.
        """
        selected = {
            value.removeprefix("system:")
            for value in controllers
            if is_system_controller(value)
        }
        if not selected.issubset(declarations):
            raise ValueError("Choose a System declared when DevClient started")
        old = self.active

        def load(candidate: SystemInstallation) -> DebuggerSession:
            """Acquire selected factory resources before initializing their memory."""
            methods: dict[str, System] = {}
            for alias in sorted(selected):
                candidate.check_cancelled()
                loaded = load_factory(declarations[alias])
                if old is not None and any(
                    loaded is value for value in old.loaded.values()
                ):
                    raise ValueError(
                        "A factory must return a fresh System for replacement"
                    )
                candidate.loaded[alias] = loaded
                method = loaded if isinstance(loaded, System) else shared_policy(loaded)
                methods[alias] = method
                if method.resource_scope is not None:
                    candidate.resources.enter_context(method.resource_scope(False))
            pair = tuple(
                methods[value.removeprefix("system:")]
                if is_system_controller(value)
                else None
                for value in controllers
            )
            candidate.check_cancelled()
            return build((pair[0], pair[1]))

        return self._prepare_owned(base, load, directory=directory)

    def prepare_loaded(
        self,
        base: DebuggerSession,
        load: Callable[[], LoadedControllers],
        build: Callable[[LoadedControllers], DebuggerSession],
        *,
        directory: Path | None,
    ) -> PendingSystemOperation:
        """Resolve a validated private setup through the shared loader and worker.

        load is a trusted host callback; it must accept only already validated
        built-ins, checkpoint Paths or launcher-declared factories. build receives
        the exact resolved assignments and initializes the new session. Both run
        outside HTTP locks in this replacement's owned context. directory follows
        prepare. Failed loading leaves the accepted game and its methods intact.
        """
        old = self.active

        def loaded_session(candidate: SystemInstallation) -> DebuggerSession:
            """Lease distinct resolved methods once, without inspecting composition."""
            resolved = load()
            originals = resolved.original_methods or resolved.methods
            for index, method in enumerate(originals):
                candidate.check_cancelled()
                if old is not None and any(
                    method is value for value in old.loaded.values()
                ):
                    raise ValueError(
                        "A replacement must load fresh methods; "
                        "use Restart to reuse them"
                    )
                candidate.loaded[str(index)] = method
            for method in resolved.methods:
                candidate.check_cancelled()
                if method.resource_scope is not None:
                    candidate.resources.enter_context(method.resource_scope(False))
            candidate.check_cancelled()
            return build(resolved)

        return self._prepare_owned(base, loaded_session, directory=directory)

    def decide(
        self,
        base: DebuggerSession,
        dispatch: Callable[[], InputDispatchResult],
    ) -> PendingSystemOperation:
        """Prepare one complete joint step and keep its evidence outcome open.

        base must belong to the active installation. dispatch runs the existing
        input/control path once. Only the later service adoption settles its
        model answers as played. A thrown error abandons them on this context.
        Legacy Manual/Random matches have no System environment wrapper and need
        no host-model decision scope; they still use this same queued step owner.
        """
        self._idle()
        installation = self.active
        if installation is None or installation.closed or self.failed:
            raise RuntimeError("No active System installation")

        def execute() -> InputDispatchResult:
            """Stage one result with evidence for the accepted decision epoch."""
            evidence = installation.evidence
            assert evidence is not None
            installation.check_cancelled()
            cleanup = ExitStack()
            installation.decision = cleanup
            try:
                self._enter_resources(installation, cleanup, base)
                if base.environment_state is not None:
                    cleanup.enter_context(evidence.decision(base.environment_state))
                result = dispatch()
                for leaf in jax.tree.leaves(result.session.state):
                    if isinstance(leaf, jax.Array):
                        leaf.block_until_ready()
                return result
            except BaseException:
                if self.caller_thread:
                    self.failed = True
                installation.settle(False)
                raise

        future = self._submit(installation, execute)
        operation = PendingSystemOperation("decision", base, installation, future)
        self.operation = operation
        return operation

    def accept(
        self, operation: PendingSystemOperation, session: DebuggerSession
    ) -> None:
        """Queue played evidence and cleanup for an already installed result.

        The service calls this immediately at either real adoption site, before
        any later replay save or response step. Metadata/cleanup errors propagate
        through finish_accept without rewriting the accepted action as abandoned.
        """
        if self.operation is not operation or operation.cancelled or operation.adopted:
            raise RuntimeError("System operation is no longer current")
        operation.adopted = True
        installation = operation.installation
        old = self.active
        operation.replaced = old

        def accepted() -> None:
            """Flush accepted evidence and close a replaced installation in order."""
            if operation.kind == "decision":
                installation.settle(True)
                installation.metadata["accepted_turns"] = (
                    session.current_evaluation_frame.frame_index
                )
                installation.write_policy_trace(session)
                installation.write_details()
            if old is not None and old is not installation:
                if operation.reuses_methods:
                    installation.resources = old.resources.pop_all()
                    installation.loaded = old.loaded
                if old.context is installation.context:
                    binding = (
                        nullcontext() if old.evidence is None else old.evidence.bind()
                    )
                    with binding:
                        old.close()
                else:
                    old.run(old.close)

        self.active = installation
        operation.acceptance_future = self._submit(installation, accepted)

    def finish_accept(self, operation: PendingSystemOperation) -> bool:
        """Release an adopted slot after evidence and old cleanup finish.

        Never wait on arbitrary user cleanup in the service thread. Errors keep
        both installations owned and block new work instead of hiding lost cleanup.
        """
        future = operation.acceptance_future
        if future is None or not future.done():
            return False
        future.result()
        if self.operation is operation:
            self.operation = None
        return True

    def cancel(self, operation: PendingSystemOperation) -> None:
        """Fence a result now; queue its owned cleanup after late work returns.

        A cancelled fresh setup leaves the old active installation usable. A
        cancelled decision or reused-method restart makes its methods uncertain
        and requires a fresh client launch. The last accepted game/replay prefix
        remains available for inspection and saving.
        """
        if self.operation is not operation or operation.adopted:
            return
        if operation.cancelled:
            return
        operation.cancelled = True
        operation.installation.cancellation.set()
        installation = operation.installation
        if (
            self.caller_thread
            or operation.reuses_methods
            or operation.kind == "decision"
        ):
            self.failed = True
        operation.cleanup_future = self._submit(installation, installation.close)

    def finish_cancel(self, operation: PendingSystemOperation) -> bool:
        """Clear a cancelled slot only after cleanup succeeds; never wait here."""
        future = operation.cleanup_future
        if future is None or not future.done():
            return False
        future.result()
        if self.active is operation.installation:
            self.active = None
        if self.operation is operation:
            self.operation = None
        return True

    def record_saved_prefix(
        self,
        *,
        installation: SystemInstallation,
        replay_id: str,
        path: Path,
        transitions: int,
    ) -> None:
        """Record the recorder's verified saved prefix without claiming game completion.

        Only call after the recorder has verified its published bundle. Repeated
        identical acknowledgements do nothing. A new Save As path remains linked
        to the same adjacent call evidence; existing files are never moved.
        """
        if installation.directory is None:
            return
        identity = (replay_id, str(path), transitions)
        with self._queue_lock:
            if identity in installation.saved_prefix_requests:
                return
            installation.saved_prefix_requests.add(identity)

        def publish() -> None:
            """Flush calls, then add the verified replay identity atomically."""
            assert installation.evidence is not None
            if not installation.closed:
                installation.evidence.flush()
            prefixes = installation.metadata["saved_prefixes"]
            assert isinstance(prefixes, list)
            value = {
                "replay_id": replay_id,
                "path": str(path),
                "transitions": transitions,
            }
            if value not in prefixes:
                prefixes.append(value)
                installation.write_details()

        self._prune_successes()
        future = self._submit(installation, publish)
        future.add_done_callback(self._report_failure)
        with self._queue_lock:
            self.recording_futures.append(future)

    def close(self) -> None:
        """Fence pending work and queue all owned cleanup, without stopping a server.

        Do not promise to stop arbitrary factory/Python code: a running callback
        can delay cleanup and Python process exit. Ordinary HTTP work keeps its
        Client deadlines. No new work may be submitted after this call.
        """
        if self._closed:
            return
        self._prune_successes()
        if self.operation is not None:
            self.cancel(self.operation)
        active = self.active
        if active is not None and (
            self.operation is None
            or self.operation.adopted
            or active is not self.operation.installation
        ):
            self.shutdown_futures.append(self._submit(active, active.close))
        for future in self.shutdown_futures:
            future.add_done_callback(self._report_failure)
        if self.operation is not None:
            for future in (
                self.operation.cleanup_future,
                self.operation.acceptance_future,
            ):
                if future is not None:
                    future.add_done_callback(self._report_failure)
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=False)
