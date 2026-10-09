"""Check the public BattleClient launch boundary and packaged resource lifetime.

Root imports stay lazy and expose a callable in either import order. Invalid
launch options fail before setup or serving; valid ones, spawn_mode included,
reach setup unchanged. Browser assets remain alive for the server lifetime,
cleanup runs on errors, and recordings keep installed identity.
The public call opens in Agent POV and has no view argument; the private
_launch_setup takes view_mode (default Agent POV) so the browser test harness
and the DevClient can keep Oracle View. Every launch compiles the first turn on
its own session exactly once, before the service is built and the page served.
"""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from marl_battlegrounds._battle_client.model import DebuggerScenario
    from marl_battlegrounds._battle_client.setup import BattleSetup


def test_root_import_does_not_load_jax_or_clients(tmp_path: Path) -> None:
    source = """
import sys
import marl_battlegrounds as marl_bgs
assert 'battle_client' in marl_bgs.__all__
for prefix in ('jax', 'marl_battlegrounds._battle_client', 'marl_battlegrounds.viewer'):
    loaded = any(n == prefix or n.startswith(prefix + '.') for n in sys.modules)
    assert not loaded, prefix
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("implementation_first", [False, True])
def test_public_callable_survives_internal_import_order(
    tmp_path: Path, implementation_first: bool
) -> None:
    source = f"""
import importlib
import marl_battlegrounds as marl_bgs
if {implementation_first!r}:
    importlib.import_module('marl_battlegrounds._battle_client.launch')
first = marl_bgs.battle_client
assert callable(first)
for name in ('launch', 'setup', 'control', 'recording_coordinator'):
    importlib.import_module('marl_battlegrounds._battle_client.' + name)
assert marl_bgs.battle_client is first
implementation = importlib.import_module('marl_battlegrounds._battle_client.launch')
assert implementation.battle_client is first
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "options",
    [
        {"port": True},
        {"port": -1},
        {"port": 65536},
        {"port": 1.0},
        {"port": "0"},
        {"open_browser": 0},
        {"open_browser": "false"},
        {"capture_replay": 1},
        {"capture_replay": None},
        {"output_dir": True},
        {"output_dir": 12},
        {"output_dir": None},
    ],
)
def test_invalid_launch_options_fail_before_setup(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, object]
) -> None:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")

    def forbidden_setup(**_kwargs: object) -> None:
        raise AssertionError("Invalid launch options must fail before setup.")

    monkeypatch.setattr(launch, "prepare_battle_setup", forbidden_setup)
    call: Callable[..., int] = launch.battle_client
    with pytest.raises((TypeError, ValueError)):
        call(map_id=0, **options)


@pytest.mark.parametrize("ancestor_is_file", [False, True])
def test_output_file_paths_fail_before_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ancestor_is_file: bool
) -> None:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    existing = tmp_path / "existing.txt"
    existing.write_text("Preserve this file.", encoding="utf-8")
    output = existing / "recordings" if ancestor_is_file else existing

    def forbidden_setup(**_kwargs: object) -> None:
        raise AssertionError("Invalid output paths must fail before setup.")

    monkeypatch.setattr(launch, "prepare_battle_setup", forbidden_setup)
    with pytest.raises((ValueError, OSError)):
        launch.battle_client(map_id=0, output_dir=output)
    assert existing.read_text(encoding="utf-8") == "Preserve this file."


def test_public_signature_has_only_approved_keyword_options() -> None:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    from marl_battlegrounds.evaluation.evaluation_conditions import OMITTED

    parameters = inspect.signature(launch.battle_client).parameters
    omitted = {
        "map_id",
        "scenario_id",
        "team_b",
        "team_a_roster",
        "team_b_roster",
        "seed",
        "max_steps",
        "score_threshold",
        "red_zone_depth",
        "spawn_mode",
    }
    defaults = {
        "team_a": "manual",
        "port": 0,
        "replay_port": 0,
        "open_browser": True,
        "output_dir": "./recordings",
        "capture_replay": True,
    }
    assert set(parameters) == omitted | defaults.keys()
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
    for name in omitted:
        assert parameters[name].default is OMITTED, name
    for name, expected in defaults.items():
        assert parameters[name].default == expected, name


@pytest.mark.parametrize("port", [0, 65535])
def test_valid_launch_options_reach_one_prepared_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, port: int
) -> None:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    marker = object()
    setup_calls: list[dict[str, object]] = []
    serve_calls: list[dict[str, object]] = []

    def prepare(**kwargs: object) -> object:
        setup_calls.append(kwargs)
        return marker

    def serve(setup: object, **kwargs: object) -> int:
        assert setup is marker
        serve_calls.append(kwargs)
        return 23

    monkeypatch.setattr(launch, "prepare_battle_setup", prepare)
    monkeypatch.setattr(launch, "_launch_setup", serve)
    result = launch.battle_client(
        map_id=0,
        spawn_mode="swapped",
        port=port,
        open_browser=False,
        output_dir=tmp_path / "records",
        capture_replay=False,
    )
    assert result == 23
    assert len(setup_calls) == len(serve_calls) == 1
    assert setup_calls[0]["map_id"] == 0
    assert setup_calls[0]["team_a"] == "manual"
    assert setup_calls[0]["spawn_mode"] == "swapped"
    assert serve_calls[0]["port"] == port
    assert serve_calls[0]["open_browser"] is False
    assert serve_calls[0]["capture_replay"] is False
    # The public call never chooses a view: _launch_setup's Agent POV default holds.
    assert "view_mode" not in serve_calls[0]
    output = serve_calls[0]["output_dir"]
    assert isinstance(output, (str, Path))
    assert Path(output).resolve() == (tmp_path / "records").resolve()


def _stub_launch_session(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[BattleSetup, list[str]]:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    from marl_battlegrounds._battle_client.setup import BattleSetup
    from marl_battlegrounds.evaluation.models import CodeRevisionV2

    events: list[str] = []
    session = SimpleNamespace(systems=None)
    revision = CodeRevisionV2(package_version="9.8.7")
    specification = object()

    def build_specification(**kwargs: object) -> object:
        assert kwargs["code_revision"] is revision
        assert kwargs["capture_profile"] == "debug"
        return specification

    def create_session(_scenario: object, **kwargs: object) -> object:
        assert kwargs["evaluation_launch_specification"] is specification
        return session

    class Service:
        def __init__(self, received: object, **kwargs: object) -> None:
            assert received is session
            assert kwargs["recorder"] is None
            events.append("service")

        def pump(self) -> bool:
            return False

        def close(self) -> None:
            events.append("close")

    def forbidden_recording(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Disabled recording must skip recording setup.")

    def warm(received: object) -> bool:
        # The launch compiles the first turn on its own session, once, before
        # the service exists, so the page opens with nothing left to compile.
        assert received is session
        events.append("warm")
        return True

    monkeypatch.setattr(launch, "discover_code_revision_v2", lambda: revision)
    monkeypatch.setattr(
        launch, "build_debugger_evaluation_launch_specification_v1", build_specification
    )
    monkeypatch.setattr(launch, "create_session", create_session)
    monkeypatch.setattr(launch, "warm_compiled_turn", warm)
    monkeypatch.setattr(launch, "DebuggerService", Service)
    monkeypatch.setattr(launch, "capture_runtime_provenance", forbidden_recording)
    monkeypatch.setattr(launch, "preflight_replay_destination", forbidden_recording)
    monkeypatch.setattr(launch, "DebuggerReplayRecorder", forbidden_recording)
    from marl_battlegrounds._battle_client.methods import LoadedControllers

    def resolve(_setup: BattleSetup) -> LoadedControllers:
        return LoadedControllers(
            ("manual", "manual"), None, (), tuple(range(10)), ("Manual",) * 10
        )

    monkeypatch.setattr(launch, "resolve_controllers", resolve)
    setup = BattleSetup(cast("DebuggerScenario", object()), 7, "manual", "manual")
    return setup, events


@pytest.mark.parametrize(
    "failure", [None, OSError("Cannot bind port"), KeyboardInterrupt()]
)
def test_packaged_assets_live_until_server_exit_and_cleanup_always_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException | None
) -> None:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    setup, events = _stub_launch_session(monkeypatch)
    package = tmp_path / "installed-viewer"
    active = False

    def package_files(name: str) -> Path:
        assert name == "marl_battlegrounds.viewer"
        return package

    @contextmanager
    def scoped_resource(resource: object) -> Generator[Path]:
        nonlocal active
        assert resource == package / "web"
        active = True
        events.append("resource")
        try:
            yield tmp_path / "extracted-web"
        finally:
            active = False
            events.append("release")

    def serve(_service: object, **kwargs: object) -> int:
        assert active
        assert kwargs["asset_root"] == tmp_path / "extracted-web"
        assert kwargs["port"] == 0
        assert kwargs["open_browser"] is False
        assert kwargs["coordinator_router"] is None
        assert kwargs["graceful_close"] is None
        events.append("serve")
        if failure is not None:
            raise failure
        return 17

    monkeypatch.setattr(launch, "files", package_files)
    monkeypatch.setattr(launch, "as_file", scoped_resource)
    monkeypatch.setattr(launch, "serve_browser_debugger", serve)
    output = tmp_path / "untouched-recordings"
    call: Callable[..., int] = launch._launch_setup
    if failure is None:
        assert (
            call(
                setup,
                port=0,
                open_browser=False,
                output_dir=output,
                capture_replay=False,
            )
            == 17
        )
    else:
        with pytest.raises(type(failure)) as error:
            call(
                setup,
                port=0,
                open_browser=False,
                output_dir=output,
                capture_replay=False,
            )
        assert error.value is failure
    assert not active
    assert not output.exists()
    assert events == [
        "warm",
        "service",
        "resource",
        "serve",
        "release",
        "close",
    ]


@pytest.mark.parametrize(
    ("view_mode", "expected"),
    ((None, "pov"), ("pov", "pov"), ("researcher", "researcher")),
)
def test_launch_opens_in_agent_pov_unless_a_private_caller_keeps_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    view_mode: str | None,
    expected: str,
) -> None:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    setup, events = _stub_launch_session(monkeypatch)
    stub = launch.DebuggerService
    views: list[object] = []

    def capture(received: object, **kwargs: object) -> object:
        views.append(kwargs["view_mode"])
        return stub(received, **kwargs)

    monkeypatch.setattr(launch, "DebuggerService", capture)

    def serve(*_args: object, **_kwargs: object) -> int:
        return 5

    monkeypatch.setattr(launch, "serve_browser_debugger", serve)
    call: Callable[..., int] = launch._launch_setup
    extra = {} if view_mode is None else {"view_mode": view_mode}
    assert (
        call(
            setup,
            port=0,
            open_browser=False,
            output_dir=tmp_path,
            capture_replay=False,
            asset_root=tmp_path,
            **extra,
        )
        == 5
    )
    assert views == [expected]
    assert events == ["warm", "service", "close"]
    assert inspect.signature(launch._launch_setup).parameters["view_mode"].default == (
        "pov"
    )
    assert "view_mode" not in inspect.signature(launch.battle_client).parameters
    documentation = inspect.getdoc(launch.battle_client) or ""
    assert "opens in Agent POV" in documentation
    assert "Open Replay always" in documentation


@pytest.mark.parametrize("failure_at", ["files", "as_file", "enter"])
def test_asset_resolution_errors_close_the_created_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    setup, events = _stub_launch_session(monkeypatch)
    failure = OSError("Package browser assets are missing")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    @contextmanager
    def fail_enter(_resource: object) -> Generator[Path]:
        raise failure
        yield tmp_path

    def forbidden_server(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Unreadable package assets must not start serving.")

    def package_files(_name: str) -> Path:
        return tmp_path

    monkeypatch.setattr(
        launch, "files", fail if failure_at == "files" else package_files
    )
    if failure_at != "files":
        monkeypatch.setattr(
            launch, "as_file", fail if failure_at == "as_file" else fail_enter
        )
    monkeypatch.setattr(launch, "serve_browser_debugger", forbidden_server)
    with pytest.raises(OSError) as error:
        launch._launch_setup(
            setup, port=0, open_browser=False, output_dir=tmp_path, capture_replay=False
        )
    assert error.value is failure
    assert events == ["warm", "service", "close"]


def test_missing_packaged_assets_never_use_checkout_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    from marl_battlegrounds.viewer.server import build_static_manifest

    setup, events = _stub_launch_session(monkeypatch)
    package = tmp_path / "installed-viewer"
    package.mkdir()
    checked_roots: list[Path] = []

    def serve(_service: object, **kwargs: object) -> int:
        root = kwargs["asset_root"]
        assert isinstance(root, Path)
        checked_roots.append(root)
        build_static_manifest(root)
        raise AssertionError("Missing package files cannot produce a server.")

    def package_files(_name: str) -> Path:
        return package

    monkeypatch.setattr(launch, "files", package_files)
    monkeypatch.setattr(launch, "serve_browser_debugger", serve)
    with pytest.raises(OSError):
        launch._launch_setup(
            setup, port=0, open_browser=False, output_dir=tmp_path, capture_replay=False
        )
    assert checked_roots == [package / "web"]
    assert events == ["warm", "service", "close"]


def test_explicit_internal_asset_root_skips_package_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    setup, events = _stub_launch_session(monkeypatch)

    def forbidden_resources(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("An explicit internal asset root needs no package lookup.")

    def serve(_service: object, **kwargs: object) -> int:
        assert kwargs["asset_root"] is tmp_path
        return 7

    monkeypatch.setattr(launch, "files", forbidden_resources)
    monkeypatch.setattr(launch, "as_file", forbidden_resources)
    monkeypatch.setattr(launch, "serve_browser_debugger", serve)
    assert (
        launch._launch_setup(
            setup,
            port=0,
            open_browser=False,
            output_dir=tmp_path,
            capture_replay=False,
            asset_root=tmp_path,
        )
        == 7
    )
    assert events == ["warm", "service", "close"]


def test_default_capture_saves_frame_zero_with_installed_v2_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = importlib.import_module("marl_battlegrounds._battle_client.launch")
    from marl_battlegrounds._battle_client.protocol import (
        CommandRequestV1,
        KeyboardCommandV1,
    )
    from marl_battlegrounds._battle_client.service import DebuggerService
    from marl_battlegrounds.evaluation.models import CodeRevisionV2
    from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
    from marl_battlegrounds.evaluation.replay_io import load_replay

    revision = CodeRevisionV2(package_version="9.8.7")
    runtime = RuntimeProvenanceV1(
        python_version="3.14.0",
        package_version=revision.package_version,
        jax_version="0.7.0",
        jaxlib_version="0.7.0",
        numpy_version="2.3.0",
        pydantic_version="2.11.0",
        platform="linux",
        machine="x86_64",
        backend="cpu",
        device="test-cpu",
        precision="float32",
        environment_count=1,
        batch_shape=(1,),
        policy_execution_included=False,
    )
    captures: list[tuple[str, bool]] = []
    package = tmp_path / "installed-viewer"
    assets = Path(__file__).resolve().parents[2] / "web" / "visual_debugger"
    active = False

    def capture(
        version: str, *, policy_execution_included: bool
    ) -> RuntimeProvenanceV1:
        captures.append((version, policy_execution_included))
        return runtime

    def package_files(name: str) -> Path:
        assert name == "marl_battlegrounds.viewer"
        return package

    @contextmanager
    def scoped_resource(resource: object) -> Generator[Path]:
        nonlocal active
        assert resource == package / "web"
        active = True
        try:
            yield assets
        finally:
            active = False

    def serve(service: DebuggerService, **kwargs: object) -> int:
        assert active
        assert kwargs["asset_root"] == assets
        assert kwargs["port"] == 0
        assert kwargs["open_browser"] is False
        assert kwargs["coordinator_router"] is not None
        assert callable(kwargs["graceful_close"])
        assert service.session.evaluation_context.code_revision == revision
        status = service.recording_status
        assert status is not None
        assert status.captured_transition_count == 0
        output = tmp_path / "recordings"
        assert output.is_dir()
        assert not list(output.glob("*.marlbg-replay.json"))
        # A closeout with no played tick saves nothing, so one tick is played.
        service.apply_command(
            CommandRequestV1(
                client_id="launch-test",
                command_id="one-turn",
                base_revision=service.revision,
                command=KeyboardCommandV1(key="Enter"),
            )
        )
        assert service.close_recording_for_keyboard_interrupt().saved
        return 19

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launch, "discover_code_revision_v2", lambda: revision)
    monkeypatch.setattr(launch, "capture_runtime_provenance", capture)
    monkeypatch.setattr(launch, "files", package_files)
    monkeypatch.setattr(launch, "as_file", scoped_resource)
    monkeypatch.setattr(launch, "serve_browser_debugger", serve)
    assert (
        launch.battle_client(
            map_id=0,
            team_a_roster=["warrior"],
            team_b_roster=["warrior"],
            max_steps=2,
            open_browser=False,
        )
        == 19
    )
    assert not active
    assert captures == [(revision.package_version, False)]
    saved = list((tmp_path / "recordings").glob("*.marlbg-replay.json"))
    assert len(saved) == 1
    replay = load_replay(saved[0]).replay
    assert type(replay.header.context.code_revision) is CodeRevisionV2
    assert replay.header.context.code_revision == revision
    assert replay.header.context.code_revision.commit_sha is None
    assert replay.header.context.code_revision.source_tree_digest is None
    assert replay.header.context.code_revision.is_dirty is None
    assert replay.header.runtime_provenance == runtime
    assert len(replay.frames) == 2
    assert len(replay.transitions) == 1


@pytest.mark.parametrize("raises_error", [False, True])
def test_browser_open_failure_keeps_the_printed_url_and_server_alive(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raises_error: bool,
) -> None:
    from marl_battlegrounds.viewer import server

    events: list[str] = []
    url = "http://127.0.0.1:12345/#token=test-launch-token"

    class Coordinator:
        product_kind = "combat_debugger"

    class FakeServer:
        expected_origin = "http://127.0.0.1:12345"
        capability_token = "test-launch-token"
        coordinator = Coordinator()

        def __enter__(self) -> FakeServer:
            events.append("enter")
            return self

        def __exit__(
            self, _error_type: object, _error: object, _traceback: object
        ) -> None:
            events.append("close")

        def coordinator_snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(service=object())

        def serve_forever(self, *, poll_interval: float) -> None:
            assert poll_interval > 0
            events.append("serve")

    def browser_open(received: str) -> bool:
        assert received == url
        events.append("browser")
        if raises_error:
            raise OSError("No desktop browser")
        return False

    monkeypatch.setattr(server.webbrowser, "open", browser_open)
    assert (
        server.serve_debugger_server(
            cast("server.DebuggerHTTPServer", FakeServer()), open_browser=True
        )
        == 0
    )
    captured = capsys.readouterr()
    assert f"MARL-BattleGrounds BattleClient: {url}" in captured.out
    assert "browser could not be opened" in captured.err
    assert "use the URL above" in captured.err
    assert events == ["enter", "browser", "serve", "close"]
