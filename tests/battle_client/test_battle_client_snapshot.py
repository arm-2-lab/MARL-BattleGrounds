"""Check atomic live HTTP reads, conditional polling and server-owned progress.

Snapshot reads retain the existing token, loopback-origin, body and method checks.
Unchanged revisions return 304 without display building or JSON serialization.
The real server loop advances Run on its launch thread without browser polling.
"""

import json

# pyright: reportPrivateUsage=false
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from threading import Thread, get_ident
from time import sleep

import jax
import pytest
from tests.battle_client.test_battle_client_run import _autonomous
from tests.battle_client.test_battle_client_worker import _method
from tests.visual_debugger.test_visual_debugger_recording_coordinator import (
    _coordinator_and_recorder,
)
from tests.visual_debugger.test_visual_debugger_server import (
    _ASSET_ROOT,
    _TOKEN,
    _exchange,
    _service,
)
from tests.visual_debugger.test_visual_debugger_systems import _send

from marl_battlegrounds._battle_client.protocol import RunCommandV1
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.viewer.server import (
    DebuggerHTTPServer,
    create_server,
    serve_debugger_server,
)


@contextmanager
def _server(service: DebuggerService) -> Generator[DebuggerHTTPServer]:
    server = create_server(
        service, asset_root=_ASSET_ROOT, port=0, capability_token=_TOKEN
    )
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _headers(server: DebuggerHTTPServer) -> dict[str, str]:
    return {
        "X-MARL-Debugger-Token": _TOKEN,
        "Origin": f"http://127.0.0.1:{server.server_port}",
    }


def test_atomic_snapshot_matches_existing_reads_and_etag_skips_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    calls = 0
    original = service.current_snapshot

    def snapshot() -> object:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(service, "current_snapshot", snapshot)
    try:
        with _server(service) as server:
            headers = _headers(server)
            response, body = _exchange(
                server, "GET", "/api/live/snapshot", headers=headers
            )
            assert response.status == HTTPStatus.OK
            payload = json.loads(body)
            assert payload["frame"] == service.current_frame().model_dump(mode="json")
            assert payload[
                "presentation"
            ] == service.current_presentation().payload.model_dump(mode="json")
            etag = f'"{payload["frame"]["session_id"]}:{payload["frame"]["revision"]}"'
            assert response.getheader("ETag") == etag
            assert response.getheader("Cache-Control") == "no-store"
            assert calls == 1
            for _ in range(3):
                response, body = _exchange(
                    server,
                    "GET",
                    "/api/live/snapshot",
                    headers={**headers, "If-None-Match": etag},
                )
                assert response.status == HTTPStatus.NOT_MODIFIED
                assert body == b""
                assert response.getheader("ETag") == etag
            assert calls == 1
            response, body = _exchange(
                server,
                "GET",
                "/api/live/snapshot",
                headers={**headers, "If-None-Match": '"other:0"'},
            )
            assert response.status == HTTPStatus.OK
            assert calls == 2
    finally:
        service.close()


@pytest.mark.parametrize(
    "case,status",
    [
        ("missing_token", 401),
        ("bad_token", 401),
        ("foreign_origin", 403),
        ("body", 400),
        ("transfer", 400),
        ("post", 405),
        ("put", 405),
    ],
)
def test_snapshot_security_rejects_before_build(
    monkeypatch: pytest.MonkeyPatch, case: str, status: int
) -> None:
    service = _service()

    def forbidden() -> object:
        raise AssertionError("Rejected snapshots must not build display data")

    monkeypatch.setattr(service, "current_snapshot", forbidden)
    try:
        with _server(service) as server:
            headers = _headers(server)
            body = None
            method = "GET"
            if case == "missing_token":
                headers.pop("X-MARL-Debugger-Token")
            elif case == "bad_token":
                headers["X-MARL-Debugger-Token"] = "incorrect"
            elif case == "foreign_origin":
                headers["Origin"] = "http://evil.example"
            elif case == "body":
                body = b"{}"
            elif case == "transfer":
                headers["Transfer-Encoding"] = "chunked"
            else:
                method = case.upper()
                body = b"{}"
            response, data = _exchange(
                server, method, "/api/live/snapshot", headers=headers, body=body
            )
            assert response.status == status, data
            assert json.loads(data)["latest_frame"] is None
    finally:
        service.close()


def test_real_server_pumps_without_browser_polling_in_original_thread() -> None:
    calls: list[str] = []
    method = _method(calls)
    original_apply = method.apply
    owner = get_ident()
    seen: list[int] = []

    def apply(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> SystemOutput:
        seen.append(get_ident())
        return original_apply(variables, memory, inputs, key)

    service, _worker = _autonomous(replace(method, apply=apply))
    server = create_server(
        service, asset_root=_ASSET_ROOT, port=0, capability_token=_TOKEN
    )
    _send(service, RunCommandV1())
    errors: list[BaseException] = []

    def stop_after_real_turns() -> None:
        try:
            while service.session.current_evaluation_frame.frame_index < 2:
                sleep(0.002)
            headers = {**_headers(server), "Content-Type": "application/json"}
            frame = service.current_frame()
            pause = {
                "schema_version": 1,
                "client_id": "second-tab",
                "command_id": "pause",
                "base_revision": 0,
                "command": {
                    "command_type": "pause",
                    "session_id": frame.session_id,
                    "run_generation": frame.run_generation,
                },
            }
            response, body = _exchange(
                server,
                "POST",
                "/api/command",
                headers=headers,
                body=json.dumps(pause).encode(),
            )
            assert response.status == 200, body
            while service.current_frame().system_operation is not None:
                sleep(0.002)
            count = service.session.current_evaluation_frame.frame_index
            sleep(0.02)
            assert service.session.current_evaluation_frame.frame_index == count
            request = {
                **pause,
                "command_id": "exit",
                "base_revision": service.revision,
                "command": {"command_type": "exit"},
            }
            response, body = _exchange(
                server,
                "POST",
                "/api/command",
                headers=headers,
                body=json.dumps(request).encode(),
            )
            assert response.status == 200, body
        except BaseException as exc:
            errors.append(exc)
            server.shutdown()

    thread = Thread(target=stop_after_real_turns)
    thread.start()
    try:
        assert serve_debugger_server(server, open_browser=False) == 0
        thread.join()
        assert not errors
        assert seen and all(identity == owner for identity in seen)
        assert calls.count("apply") >= 4
    finally:
        service.close()
        thread.join()


def test_recording_coordinator_exposes_the_same_atomic_snapshot(tmp_path: Path) -> None:
    coordinator, recorder = _coordinator_and_recorder(tmp_path)
    service = coordinator.service
    server = create_server(
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator_router=coordinator.router,
    )
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        response, body = _exchange(
            server, "GET", "/api/live/snapshot", headers=_headers(server)
        )
        assert response.status == 200
        payload = json.loads(body)
        assert payload["frame"]["recording"]["lifecycle"] == "recording"
        assert (
            payload["frame"]["revision"]
            == payload["presentation"]["source"]["source_revision"]
        )
        assert payload["frame"]["frame_index"] == 0
        assert recorder.validated_transition_count == 0
        assert coordinator.router.snapshot().binding.current_snapshot is not None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        service.close()
