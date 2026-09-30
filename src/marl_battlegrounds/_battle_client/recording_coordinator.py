"""Keep live recording and a separate saved Replay Viewer under one lifetime.

The live HTTP router never changes. Explicit Open Replay requests start or reuse
one child Viewer with public assets and its own loopback token. The launcher
closes this coordinator before releasing assets; recording and persistence stay
with the live service. Browser tabs do not own either server's lifetime.
"""

from __future__ import annotations

from dataclasses import replace
from http.client import HTTPConnection
from pathlib import Path
from threading import Lock, Thread

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
)


class RecordingDebuggerCoordinator:
    """Own a recording-enabled live router and at most one saved Viewer.

    Parameters
    ----------
    service : DebuggerService
        Exact live service with recording enabled.
    asset_root : pathlib.Path
        Explicit public browser assets. Keep this directory available until
        close returns. A private launch must also pass the public assets here.

    Raises
    ------
    TypeError
        The service is not the exact supported type.
    ValueError
        The live service has no recording status.

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
        "_live_binding",
        "_router",
        "_service",
        "_viewer_lock",
        "_viewer_replay_id",
        "_viewer_server",
        "_viewer_thread",
    )

    def __init__(self, service: DebuggerService, *, asset_root: Path) -> None:
        """Bind the recording service and retain its explicit public asset root."""
        if type(service) is not DebuggerService:
            raise TypeError("recording coordinator requires exact DebuggerService")
        if service.recording_status is None:
            raise ValueError("recording coordinator requires recording-enabled service")
        self._service = service
        self._asset_root = asset_root
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
        command owner, which returns a path-free retry notice.
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

        candidate = create_server(
            viewer,
            asset_root=self._asset_root,
            port=0,
            coordinator=replay_http_binding(viewer),
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
        return f"{candidate.expected_origin}/#token={candidate.capability_token}"

    def apply_command(self, request: CommandRequestV1) -> ServiceCommandResult:
        """Apply a live command and open a separate Viewer only when requested.

        request is the checked live command. Return the service result with an
        optional replay_url for a ready Viewer. Startup failure instead adds a
        notice; the saved file and live router remain intact. No browser opens
        on the host: the requesting page owns its tab and popup fallback link.
        An accepted Exit closes the child before requesting parent shutdown.
        """
        result = self._service.apply_command(request)
        if result.shutdown_requested:
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


__all__ = ["RecordingDebuggerCoordinator"]
