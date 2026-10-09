"""Keep live recording and a separate saved Replay Viewer under one lifetime.

The live HTTP router never changes. Explicit Open Replay requests start or reuse
one child Viewer with public assets and its own loopback token. The child
listens on a free loopback port chosen by the operating system, or on a fixed
``replay_port`` (for example one that Docker publishes in advance). The launcher
closes this coordinator before releasing assets; recording and persistence stay
with the live service. Browser tabs do not own either server's lifetime.

With ``keep_viewer_after_exit=True`` (the BattleClient), Exit stops only the
live client: the launcher takes the running child Viewer with
``release_kept_viewer`` and keeps serving it (``KeptReplayViewer``) until the
Viewer's own Exit or Ctrl-C. The child serves its page from bytes loaded when
it started, so it needs no asset folder after that. A new child serves at once,
but holds its page until its replay service has built every agent's Agent POV
work (``ReplayViewerService.prepare_agent_pov``, run on a background thread), so
Open Replay answers at once, the BattleClient keeps playing, and the replay tab
opens with Agent POV ready.
"""

from __future__ import annotations

from dataclasses import replace
from http.client import HTTPConnection
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic

from marl_battlegrounds._battle_client.protocol import (
    ApiErrorV2,
    CommandRequestV1,
    CommandResponseV2,
)
from marl_battlegrounds._battle_client.service import (
    DebuggerService,
    ServiceCommandResult,
)
from marl_battlegrounds.viewer.launch import replay_http_binding
from marl_battlegrounds.viewer.replay_service import ReplayViewerService
from marl_battlegrounds.viewer.server import (
    LIVE_HTTP_ROUTES,
    DebuggerHTTPServer,
    GracefulCloseResult,
    HttpCoordinatorBinding,
    HttpCoordinatorRouter,
    create_server,
    wait_for_quiet_requests,
)


class _GiveWayToRequests:
    """The Open Replay prepare's pause: let the game go first, but always finish.

    Called before each small piece of ReplayViewerService.prepare_agent_pov.
    It waits until no request of this process runs (viewer.server
    .wait_for_quiet_requests), but for at most WAIT_SECONDS (0.5 s); when
    requests keep coming that long, it lets the prepare work for
    WORK_SECONDS (0.25 s) without waiting, then waits again. So with requests
    nonstop the prepare still gets about a third of the time and always
    finishes, and with gaps of 0.15 s or more it works only in the gaps.
    Measured on CPU (3 Oct 2026, a 253-tick prepare right after Restart into
    a Policies Only game on Run, whose page polls the game about 6 times a
    second): the replay tab was ready after 25 s (about 10 s with the game
    left alone); Run kept its pace and its slowest request took 457 ms.
    Waiting at most 1.0 s gave 40 s and 553 ms; 0.25 s gave 17 s and 832 ms;
    with no bound the prepare had not finished after 300 s. Plain
    monotonic-clock state; one instance per prepare, used on one thread.
    """

    WAIT_SECONDS = 0.5
    WORK_SECONDS = 0.25

    def __init__(self) -> None:
        """Start with no work slice open."""
        self._work_until = 0.0

    def __call__(self) -> None:
        """Wait for quiet requests, or open a work slice after the longest wait."""
        if monotonic() < self._work_until:
            return
        if not wait_for_quiet_requests(longest_wait_seconds=self.WAIT_SECONDS):
            self._work_until = monotonic() + self.WORK_SECONDS


def _prepare_then_release_page(viewer: ReplayViewerService, page_ready: Event) -> None:
    """Build a child Viewer's Agent POV work, then let its page open.

    Parameters
    ----------
    viewer : ReplayViewerService
        The child's replay service; ReplayViewerService.prepare_agent_pov
        runs on it (at once when already prepared; it returns False and
        leaves the rest on demand when a piece fails).
    page_ready : threading.Event
        The child server's page hold (create_server page_ready); always set
        at the end, also after an error, so the page never waits forever.

    Notes
    -----
    Runs on the coordinator's ``marl-replay-agent-pov-prepare`` daemon
    thread. Before each small piece the prepare gives way to this process's
    requests (_GiveWayToRequests), so the game goes first: without that, a
    Restart during a 253-tick prepare took 2.7 to 2.9 s instead of 0.15 s
    (CPU, 3 Oct 2026), because each of its many short JAX calls hands the
    Python lock to the busy prepare thread. The pause is bounded, so the
    prepare finishes even while requests never stop. A child stopped
    meanwhile does not stop this work; it ends with the prepare, or with the
    process.
    """
    try:
        viewer.prepare_agent_pov(between=_GiveWayToRequests())
    finally:
        page_ready.set()


class RecordingDebuggerCoordinator:
    """Own a recording-enabled live router and at most one saved Viewer.

    Parameters
    ----------
    service : DebuggerService
        Exact live service with recording enabled.
    asset_root : pathlib.Path
        Explicit public browser assets. Keep this directory available until
        close returns. A private launch must also pass the public assets here.
    keep_viewer_after_exit : bool, default=False
        True lets an opened Replay Viewer outlive the live client's Exit: Exit
        fences new opens but leaves the running child serving, and the launcher
        takes it with release_kept_viewer. False stops the child as soon as
        Exit is accepted.
    replay_port : int, default=0
        Loopback port for every child Viewer, from 0 through 65535. Zero asks
        the operating system for a free port at each start. A fixed port has
        room for one child, so opening a different replay stops the old child
        before the new one starts there; if that start then fails, no Viewer
        runs until the next Open Replay. With zero, the new child starts first
        and the old one keeps serving if the start fails.

    Raises
    ------
    TypeError
        The service is not the exact supported type, keep_viewer_after_exit
        is not a bool, or replay_port is not an int.
    ValueError
        The live service has no recording status, or replay_port is outside
        0 through 65535.

    Notes
    -----
    Construction binds live operations without opening a socket or writing data.
    Open Replay starts a child server. close is required at parent shutdown.
    Child starts, reuse, replacement and close share one lock. A closed parent
    never starts another child. Recording itself stays owned by the service.
    """

    __slots__ = (
        "_asset_root",
        "_closed",
        "_keep_viewer_after_exit",
        "_live_binding",
        "_replay_port",
        "_router",
        "_service",
        "_viewer_lock",
        "_viewer_replay_id",
        "_viewer_server",
        "_viewer_thread",
    )

    def __init__(
        self,
        service: DebuggerService,
        *,
        asset_root: Path,
        keep_viewer_after_exit: bool = False,
        replay_port: int = 0,
    ) -> None:
        """Bind the recording service and retain its explicit public asset root.

        See the class description for the parameters and errors.
        """
        if type(service) is not DebuggerService:
            raise TypeError("recording coordinator requires exact DebuggerService")
        if type(keep_viewer_after_exit) is not bool:
            raise TypeError("keep_viewer_after_exit must be a bool")
        if type(replay_port) is not int:
            raise TypeError("replay_port must be an int")
        if not 0 <= replay_port <= 65535:
            raise ValueError("replay_port must be from 0 through 65535")
        if service.recording_status is None:
            raise ValueError("recording coordinator requires recording-enabled service")
        self._service = service
        self._asset_root = asset_root
        self._keep_viewer_after_exit = keep_viewer_after_exit
        self._replay_port = replay_port
        self._viewer_lock = Lock()
        self._viewer_server: DebuggerHTTPServer | None = None
        self._viewer_thread: Thread | None = None
        self._viewer_replay_id: str | None = None
        self._closed = False
        self._live_binding = HttpCoordinatorBinding(
            mode="live",
            initial_show_ranges=service.session.show_ranges,
            routes=LIVE_HTTP_ROUTES,
            request_model=CommandRequestV1,
            error_factory=ApiErrorV2,
            current_frame=service.current_frame,
            apply_command=self.apply_command,
            current_presentation=service.current_presentation,
            current_snapshot=service.current_snapshot,
            current_metric_report=None,
        )
        self._router = HttpCoordinatorRouter(
            service=service, binding=self._live_binding
        )

    @property
    def router(self) -> HttpCoordinatorRouter:
        """Return the fixed live router to install before the parent server starts."""
        return self._router

    @property
    def service(self) -> DebuggerService:
        """Return the live service that owns recording and match operations."""
        return self._service

    @staticmethod
    def _stop_viewer(server: DebuggerHTTPServer, thread: Thread) -> None:
        """Stop and join one child; a stopped request loop needs no shutdown call.

        The caller holds the Viewer lock. A thread that failed to start has no
        identity and must not be joined. Always release the listening socket.
        """
        if thread.ident is not None:
            if thread.is_alive():
                server.shutdown()
            thread.join()
        server.server_close()

    def _open_viewer(self, viewer: ReplayViewerService) -> str:
        """Start or reuse a saved Viewer and return its loopback capability URL.

        The caller holds the Viewer lock. A short authenticated HTTP request
        checks that a candidate actually serves before replacing the prior
        Viewer. Startup has a 2-second network limit. Failures close candidate
        resources and leave any prior Viewer intact; exceptions reach the
        command owner, which returns a path-free retry notice. With a fixed
        replay_port the prior Viewer holds that port, so it stops first, and a
        failed start leaves no Viewer until the next Open Replay.

        A new child serves at once but holds its page (create_server
        page_ready) while a daemon thread named
        ``marl-replay-agent-pov-prepare`` builds every agent's Agent POV work
        (ReplayViewerService.prepare_agent_pov, see
        _prepare_then_release_page), the way the BattleClient compiles its
        first turn before its page opens. So this returns at once, the live
        game keeps answering at full speed (the prepare waits while any of
        this process's requests runs), and the requesting page's new tab,
        which shows "Opening Replay" meanwhile, opens with Agent POV ready:
        about 8 s after a 253-tick five-against-five game on CPU when the
        game is left alone, longer while it is played, well under a second
        after a short scenario. A reused child keeps its own hold, which its thread
        releases. If that work fails its validation, the page still opens
        and the failing part builds on demand, where its error shows.
        """
        if self._closed:
            raise RuntimeError("The live client has closed.")
        current = self._viewer_server
        current_thread = self._viewer_thread
        if (
            current is not None
            and current_thread is not None
            and current_thread.is_alive()
            and not current.shutdown_started.is_set()
            and not current.debugger_service.shutting_down
            and not current.debugger_service.faulted
            and self._viewer_replay_id == viewer.replay_id
        ):
            return f"{current.expected_origin}/#token={current.capability_token}"

        if (
            self._replay_port != 0
            and current is not None
            and current_thread is not None
        ):
            # A fixed port has room for one Viewer: free it before the new start.
            self._viewer_server = None
            self._viewer_thread = None
            self._viewer_replay_id = None
            self._stop_viewer(current, current_thread)
            current = None
            current_thread = None
        page_ready = Event()
        candidate = create_server(
            viewer,
            asset_root=self._asset_root,
            port=self._replay_port,
            coordinator=replay_http_binding(viewer),
            page_ready=page_ready,
        )

        def serve() -> None:
            """Serve this child and release its socket even after a loop failure."""
            try:
                candidate.serve_forever(poll_interval=0.05)
            finally:
                candidate.server_close()

        thread = Thread(target=serve, name="marl-recording-viewer", daemon=True)
        try:
            thread.start()
            connection = HTTPConnection("127.0.0.1", candidate.server_port, timeout=2)
            try:
                connection.request(
                    "GET",
                    "/api/frame",
                    headers={"X-MARL-Debugger-Token": candidate.capability_token},
                )
                response = connection.getresponse()
                response.read()
                if response.status != 200 or not thread.is_alive():
                    raise RuntimeError("The Replay Viewer did not start.")
            finally:
                connection.close()
        except BaseException:
            self._stop_viewer(candidate, thread)
            raise

        self._viewer_server = candidate
        self._viewer_thread = thread
        self._viewer_replay_id = viewer.replay_id
        if current is not None and current_thread is not None:
            self._stop_viewer(current, current_thread)
        Thread(
            target=_prepare_then_release_page,
            args=(viewer, page_ready),
            name="marl-replay-agent-pov-prepare",
            daemon=True,
        ).start()
        return f"{candidate.expected_origin}/#token={candidate.capability_token}"

    def apply_command(self, request: CommandRequestV1) -> ServiceCommandResult:
        """Apply a live command and open a separate Viewer only when requested.

        request is the checked live command. Return the service result with an
        optional replay_url for a ready Viewer. Startup failure instead adds a
        notice; the saved file and live router remain intact. No browser opens
        on the host: the requesting page owns its tab and popup fallback link.
        An accepted Exit fences new opens. Without keep_viewer_after_exit it
        also closes the child before requesting parent shutdown; with it the
        child keeps serving for release_kept_viewer.
        """
        result = self._service.apply_command(request)
        if result.shutdown_requested:
            if self._keep_viewer_after_exit:
                with self._viewer_lock:
                    self._closed = True
            else:
                self.close()
        handoff = result.replay_handoff
        if handoff is None:
            return result
        if type(handoff) is not ReplayViewerService:
            raise TypeError("recording handoff must be exact ReplayViewerService")
        if not isinstance(result.payload, CommandResponseV2):
            raise TypeError("recording handoff requires a live command response")
        with self._viewer_lock:
            try:
                url = self._open_viewer(handoff)
            except Exception:
                return replace(
                    result,
                    payload=result.payload.model_copy(
                        update={
                            "notice": (
                                "The replay is saved, but Replay Viewer "
                                "could not open. "
                                "Try Open Replay again."
                            ),
                        }
                    ),
                )
            return replace(
                result, payload=result.payload.model_copy(update={"replay_url": url})
            )

    def close(self) -> None:
        """Close the child once and prevent future opens without changing the file.

        Safe to call repeatedly or concurrently with Open Replay. This waits for
        the owned request loop, clears its references, and never closes the live
        service; the parent launcher remains that service's owner.
        """
        with self._viewer_lock:
            self._closed = True
            server, thread = self._viewer_server, self._viewer_thread
            self._viewer_server = None
            self._viewer_thread = None
            self._viewer_replay_id = None
            if server is not None and thread is not None:
                self._stop_viewer(server, thread)

    def release_kept_viewer(self) -> KeptReplayViewer | None:
        """Hand a still-serving Replay Viewer to the launcher after Exit.

        Returns
        -------
        KeptReplayViewer or None
            The running child Viewer, now owned by the caller, when
            keep_viewer_after_exit is True and a child is still serving.
            None when no Viewer was opened, the flag is False, or the child is
            already stopping or faulted (such a child is stopped here).

        Notes
        -----
        Call after the live server stopped, before close. Later opens stay
        fenced and close finds no child to stop. The caller must end the kept
        Viewer with KeptReplayViewer.serve_until_closed or close.
        """
        with self._viewer_lock:
            self._closed = True
            server, thread = self._viewer_server, self._viewer_thread
            self._viewer_server = None
            self._viewer_thread = None
            self._viewer_replay_id = None
            if server is None or thread is None:
                return None
            if (
                self._keep_viewer_after_exit
                and thread.is_alive()
                and not server.shutdown_started.is_set()
                and not server.debugger_service.shutting_down
                and not server.debugger_service.faulted
            ):
                return KeptReplayViewer(server, thread)
            self._stop_viewer(server, thread)
            return None

    def graceful_close(self) -> GracefulCloseResult:
        """Save the live prefix on Ctrl-C, close the Viewer, and report the result.

        Return exit code zero after a successful save and one after a failed
        save. The service supplies the host-only message. Always close the child
        even if the recording closeout raises; the launcher reports that error.
        """
        try:
            result = self._service.close_recording_for_keyboard_interrupt()
            return GracefulCloseResult(
                exit_code=0 if result.saved else 1, message=result.message
            )
        finally:
            self.close()


class KeptReplayViewer:
    """A saved-replay Viewer that keeps serving after the live client exits.

    Parameters
    ----------
    server : DebuggerHTTPServer
        The running child Viewer server; this object now owns it and its
        loopback socket.
    thread : threading.Thread
        The thread running that server's request loop.

    Notes
    -----
    Only RecordingDebuggerCoordinator.release_kept_viewer makes one. The
    launcher runs serve_until_closed after the live game, its methods and its
    asset folder are released, so only the Viewer's replay stays in memory.
    """

    __slots__ = ("_server", "_thread")

    def __init__(self, server: DebuggerHTTPServer, thread: Thread) -> None:
        """Take ownership of one running child Viewer and its thread."""
        self._server = server
        self._thread = thread

    @property
    def url(self) -> str:
        """Return the Viewer's loopback address with its launch token."""
        server = self._server
        return f"{server.expected_origin}/#token={server.capability_token}"

    def serve_until_closed(self) -> int:
        """Block until the Viewer's own Exit or Ctrl-C, then release its socket.

        Returns
        -------
        int
            Zero. Ctrl-C here is the ordinary way to stop the kept Viewer.

        Notes
        -----
        Prints two lines first: that the live game closed while its Replay
        Viewer is still open (with the URL), and how to stop it. Prints
        "Replay Viewer stopped." at the end. All prints flush at once.
        """
        print(
            f"The live game has closed. Its Replay Viewer is still open: {self.url}",
            flush=True,
        )
        print(
            "Exit the Replay Viewer, or press Ctrl-C here, to finish.",
            flush=True,
        )
        try:
            while self._thread.is_alive():
                self._thread.join(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
        print("Replay Viewer stopped.", flush=True)
        return 0

    def close(self) -> None:
        """Stop the Viewer if it still serves and release its socket.

        Safe to call more than once. Use it when the launcher must stop the
        Viewer without waiting (for example after a later cleanup error).
        """
        RecordingDebuggerCoordinator._stop_viewer(self._server, self._thread)  # pyright: ignore[reportPrivateUsage]


__all__ = ["KeptReplayViewer", "RecordingDebuggerCoordinator"]
