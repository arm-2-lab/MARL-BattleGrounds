"""Check which browser files the local viewer server serves, and how.

Contracts checked here:

- PNG and WebP images under ``assets/`` are served as ``image/png`` and
  ``image/webp`` with the same ``nosniff`` and content-security headers as
  every other file, byte for byte. Files of other types under ``assets/``,
  such as Markdown notes, are not served.
- Every CSS file under ``src/`` is served as ``text/css``.
- Every look stylesheet that ``styles.css`` imports is a required asset: the
  manifest refuses to build when one is missing, while an extra stylesheet
  that nothing imports is served but not required. An ``@import`` inside a
  comment is not an import. Every other ``@import`` form, and a stylesheet
  that is not UTF-8, is refused with a clear error.
- The owner's emblem images are required assets.
- Starting from ``index.html``, every stylesheet link, icon, image, script,
  CSS ``@import``, CSS ``url()``, JavaScript module import and absolute
  ``/assets/`` or ``/src/`` path written in a script resolves to a route that
  the running server answers with the expected content type.
- The Python wheel and the source archive ship every served file at the path
  the packaged viewer reads it from.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
import tomllib
from collections.abc import Iterator
from html.parser import HTMLParser
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Literal
from urllib.parse import urljoin, urlsplit

import pytest
from pydantic import BaseModel

import marl_battlegrounds.viewer.server as server_module
from marl_battlegrounds.viewer.server import (
    LIVE_HTTP_ROUTES,
    DebuggerHTTPServer,
    HttpCommandResult,
    HttpCoordinatorBinding,
    HttpPayloadResult,
    build_static_manifest,
    create_server,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ASSET_ROOT = _REPOSITORY_ROOT / "web" / "visual_debugger"
_BRAND_IMAGES = (
    ("/assets/brand/marl-bgs-emblem-32.webp", "image/webp"),
    ("/assets/brand/marl-bgs-emblem-64.png", "image/png"),
    ("/assets/brand/marl-bgs-emblem-64.webp", "image/webp"),
    ("/assets/brand/marl-bgs-emblem-128.webp", "image/webp"),
)
_GENERATED_ROUTES = {"/bootstrap.js": "text/javascript"}
_JS_STATIC_IMPORT = re.compile(
    r"^\s*(?:import|export)\b[^;\"'`]*?\bfrom\s*[\"']([^\"']+)[\"']", re.MULTILINE
)
_JS_SIDE_EFFECT_IMPORT = re.compile(r"^\s*import\s*[\"']([^\"']+)[\"']", re.MULTILINE)
_JS_DYNAMIC_IMPORT = re.compile(r"\bimport\(\s*[\"']([^\"']+)[\"']\s*\)")
_JS_ABSOLUTE_ASSET = re.compile(r"[\"'](/(?:assets|src)/[^\"'?#\s]+)[\"']")
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_IMPORT = re.compile(r"@import\s+url\(\s*[\"']([^\"']+)[\"']\s*\)")
_CSS_URL = re.compile(r"url\(\s*[\"']?([^\"')]+?)[\"']?\s*\)")
_ORIGIN = "http://viewer.invalid"

type _Kind = Literal["stylesheet", "script", "binary", "any"]


class _UnusedRequest(BaseModel):
    command: str


class _NotFoundError(BaseModel):
    error_code: str
    message: str


def _no_frame() -> object:
    raise AssertionError("Static asset requests must not reach the service.")


def _no_command(*_args: object, **_kwargs: object) -> HttpCommandResult:
    raise AssertionError("Static asset requests must not reach the service.")


def _no_presentation() -> HttpPayloadResult:
    raise AssertionError("Static asset requests must not reach the service.")


def _static_only_binding() -> HttpCoordinatorBinding:
    return HttpCoordinatorBinding(
        mode="live",
        routes=LIVE_HTTP_ROUTES,
        request_model=_UnusedRequest,
        error_factory=_NotFoundError,
        current_frame=_no_frame,
        apply_command=_no_command,
        current_presentation=_no_presentation,
    )


@pytest.fixture(scope="module")
def static_server() -> Iterator[DebuggerHTTPServer]:
    server = create_server(
        object(),
        asset_root=_ASSET_ROOT,
        port=0,
        coordinator=_static_only_binding(),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _get(server: DebuggerHTTPServer, route: str) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=None)
    try:
        connection.request("GET", route)
        response = connection.getresponse()
        body = response.read()
        headers = {name.lower(): value for name, value in response.getheaders()}
        return response.status, headers, body
    finally:
        connection.close()


def _minimal_root(tmp_path: Path, styles: str | bytes) -> Path:
    root = tmp_path / "web"
    for relative in server_module._REQUIRED_RUNTIME_ASSET_PATHS:  # pyright: ignore[reportPrivateUsage]
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"/* placeholder */")
    raw = styles.encode("utf-8") if isinstance(styles, str) else styles
    (root / "styles.css").write_bytes(raw)
    return root


def _write(root: Path, relative: str, text: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _PageReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.found: list[tuple[str, _Kind]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if tag == "link" and values.get("href"):
            rel = set(values.get("rel", "").lower().split())
            kind: _Kind = "stylesheet" if "stylesheet" in rel else "binary"
            if rel & {"modulepreload"}:
                kind = "script"
            self.found.append((values["href"], kind))
        elif tag == "script" and values.get("src"):
            self.found.append((values["src"], "script"))
        elif tag in {"img", "source"}:
            if values.get("src"):
                self.found.append((values["src"], "binary"))
            for candidate in values.get("srcset", "").split(","):
                address = candidate.strip().split(" ", 1)[0]
                if address:
                    self.found.append((address, "binary"))


def _resolve(base_route: str, reference: str) -> str | None:
    if reference.startswith(("data:", "#")):
        return None
    assert not reference.startswith(("http:", "https:", "//")), reference
    assert reference.startswith(("/", "./", "../")), (
        f"{base_route} must name {reference!r} by path"
    )
    parts = urlsplit(urljoin(f"{_ORIGIN}{base_route}", reference))
    assert f"{parts.scheme}://{parts.netloc}" == _ORIGIN, reference
    return posixpath.normpath(parts.path)


def _references(route: str, body: bytes) -> list[tuple[str, _Kind]]:
    if not route.endswith((".css", ".js")):
        return []
    text = body.decode("utf-8")
    if route.endswith(".css"):
        css = _CSS_COMMENT.sub("", text)
        imports = _CSS_IMPORT.findall(css)
        other_urls = [url for url in _CSS_URL.findall(css) if url not in imports]
        return [
            *((url, "stylesheet") for url in imports),
            *((url, "binary") for url in other_urls),
        ]
    modules = [
        *_JS_STATIC_IMPORT.findall(text),
        *_JS_SIDE_EFFECT_IMPORT.findall(text),
        *_JS_DYNAMIC_IMPORT.findall(text),
    ]
    return [
        *((module, "script") for module in modules),
        *((path, "any") for path in _JS_ABSOLUTE_ASSET.findall(text)),
    ]


def _expected_type(kind: _Kind, content_type: str) -> bool:
    if kind == "stylesheet":
        return content_type.startswith("text/css")
    if kind == "script":
        return content_type.startswith("text/javascript")
    if kind == "binary":
        return content_type.startswith(("image/", "font/"))
    return True


def test_png_and_webp_are_served_with_exact_types_and_security_headers(
    static_server: DebuggerHTTPServer,
) -> None:
    for route, content_type in _BRAND_IMAGES:
        status, headers, body = _get(static_server, route)
        assert status == 200, route
        assert headers["content-type"] == content_type
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["cache-control"] == "no-store"
        policy = headers["content-security-policy"]
        assert "default-src 'self'" in policy
        assert "img-src 'self' data:" in policy
        assert "frame-ancestors 'none'" in policy
        assert body == (_ASSET_ROOT / route.removeprefix("/")).read_bytes()
        if content_type == "image/png":
            assert body.startswith(b"\x89PNG\r\n\x1a\n")
        else:
            assert body[:4] == b"RIFF"
            assert body[8:12] == b"WEBP"


def test_files_of_other_types_under_assets_are_not_served(
    static_server: DebuggerHTTPServer,
) -> None:
    for route in ("/assets/brand/PROVENANCE.md", "/assets/fonts/PROVENANCE.md"):
        assert (_ASSET_ROOT / route.removeprefix("/")).is_file()
        status, headers, _body = _get(static_server, route)
        assert status == 404
        assert headers["x-content-type-options"] == "nosniff"


def test_every_css_file_under_src_is_served_as_css(
    static_server: DebuggerHTTPServer,
) -> None:
    manifest = build_static_manifest(_ASSET_ROOT)
    stylesheets = sorted((_ASSET_ROOT / "src").rglob("*.css"))
    assert stylesheets
    for path in stylesheets:
        route = f"/{path.relative_to(_ASSET_ROOT).as_posix()}"
        assert manifest[route].content_type == "text/css; charset=utf-8"
        status, headers, body = _get(static_server, route)
        assert status == 200
        assert headers["content-type"].startswith("text/css")
        assert headers["x-content-type-options"] == "nosniff"
        assert body == path.read_bytes()


def test_brand_images_and_imported_stylesheets_are_required() -> None:
    required = set(server_module._REQUIRED_RUNTIME_ASSET_PATHS)  # pyright: ignore[reportPrivateUsage]
    for route, _content_type in _BRAND_IMAGES:
        assert route.removeprefix("/") in required
    imported = server_module._imported_stylesheet_paths(  # pyright: ignore[reportPrivateUsage]
        (_ASSET_ROOT / "styles.css").read_bytes()
    )
    assert imported
    assert all(path.startswith("src/styles/") for path in imported)
    assert imported[-1] == "src/styles/palette.css"
    manifest = build_static_manifest(_ASSET_ROOT)
    for path in imported:
        assert f"/{path}" in manifest


def test_a_missing_imported_stylesheet_stops_the_manifest(tmp_path: Path) -> None:
    root = _minimal_root(
        tmp_path,
        '@import url("/src/styles/hud.css");\n'
        '@import url("/src/styles/palette.css");\n'
        ":root { --x: 1; }\n",
    )
    _write(root, "src/styles/hud.css")
    _write(root, "src/styles/palette.css")
    _write(root, "src/styles/not-imported.css")
    manifest = build_static_manifest(root)
    for route in (
        "/src/styles/hud.css",
        "/src/styles/palette.css",
        "/src/styles/not-imported.css",
    ):
        assert manifest[route].content_type == "text/css; charset=utf-8"

    (root / "src/styles/not-imported.css").unlink()
    assert "/src/styles/not-imported.css" not in build_static_manifest(root)

    (root / "src/styles/hud.css").unlink()
    with pytest.raises(ValueError, match="required browser asset is missing") as error:
        build_static_manifest(root)
    assert str(error.value).endswith(str(root.resolve() / "src/styles/hud.css"))


def test_an_import_inside_a_comment_is_not_required(tmp_path: Path) -> None:
    root = _minimal_root(
        tmp_path,
        '/* Old: @import url("/src/styles/gone.css"); */\n:root { --x: 1; }\n',
    )
    assert "/src/styles/gone.css" not in build_static_manifest(root)


@pytest.mark.parametrize(
    "rule",
    (
        '@import "/src/styles/hud.css";',
        "@import url('/src/styles/hud.css');",
        '@import url("src/styles/hud.css");',
        '@import url("../src/styles/hud.css");',
        '@import url("/assets/hud.css");',
        '@import url("/src/styles/hud.css") layer(looks);',
        '@import url("/src/styles/hud.css") screen;',
        '@import url("https://example.com/hud.css");',
        '@IMPORT url("/src/styles/hud.css");',
        '@import url("/src/styles/Hud.css");',
    ),
)
def test_other_import_forms_are_refused(tmp_path: Path, rule: str) -> None:
    root = _minimal_root(tmp_path, f"{rule}\n:root {{ --x: 1; }}\n")
    _write(root, "src/styles/hud.css")
    with pytest.raises(
        ValueError, match=r"unsupported stylesheet import in styles\.css"
    ):
        build_static_manifest(root)


def test_a_stylesheet_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    root = _minimal_root(tmp_path, b":root { --x: '\xff'; }\n")
    with pytest.raises(ValueError, match="not UTF-8"):
        build_static_manifest(root)


def test_every_reference_reachable_from_the_page_is_served(
    static_server: DebuggerHTTPServer,
) -> None:
    page = _PageReferences()
    page.feed((_ASSET_ROOT / "index.html").read_text(encoding="utf-8"))
    pending: list[tuple[str, str, _Kind]] = [
        ("/index.html", reference, kind) for reference, kind in page.found
    ]
    checked: dict[str, _Kind] = {}
    while pending:
        base, reference, kind = pending.pop()
        route = _resolve(base, reference)
        if route is None or (route, kind) in checked.items():
            continue
        status, headers, body = _get(static_server, route)
        assert status == 200, f"{base} names {reference!r}, which is not served"
        content_type = headers["content-type"]
        assert _expected_type(kind, content_type), f"{route}: {content_type}"
        if route in checked:
            continue
        checked[route] = kind
        if route in _GENERATED_ROUTES:
            assert content_type.startswith(_GENERATED_ROUTES[route])
            continue
        pending.extend(
            (route, child, child_kind) for child, child_kind in _references(route, body)
        )
    for required in (
        "/styles.css",
        "/src/main.js",
        "/src/replay-export.js",
        "/src/styles/palette.css",
        "/assets/fonts/AtkinsonHyperlegible-Bold.woff2",
    ):
        assert required in checked, required


def test_the_wheel_and_source_archive_ship_every_served_file() -> None:
    project = tomllib.loads((_REPOSITORY_ROOT / "pyproject.toml").read_text("utf-8"))
    build = project["tool"]["hatch"]["build"]
    force_include: dict[str, str] = build["targets"]["wheel"]["force-include"]
    sdist_include: list[str] = build["targets"]["sdist"]["include"]
    excluded: list[str] = build.get("exclude", [])
    web = _ASSET_ROOT.relative_to(_REPOSITORY_ROOT).as_posix()
    manifest = build_static_manifest(_ASSET_ROOT)
    routes = sorted(route for route in manifest if route != "/")
    assert routes
    for route in routes:
        source = f"{web}{route}"
        destinations = [
            f"{target}{source.removeprefix(origin)}"
            for origin, target in force_include.items()
            if source == origin or source.startswith(f"{origin}/")
        ]
        assert destinations == [f"marl_battlegrounds/viewer/web{route}"], source
        assert any(
            f"/{source}" == entry or f"/{source}".startswith(f"{entry}/")
            for entry in sdist_include
        ), source
        assert not any(fnmatch.fnmatch(f"/{source}", pattern) for pattern in excluded)
