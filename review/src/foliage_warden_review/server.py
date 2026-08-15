"""Loopback-only HTTP service for the review UI."""

from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .manifest import Manifest, MediaItem, load_manifest
from .storage import AnnotationStore, RevisionConflict, _stable_json
from .validation import AnnotationError

MAX_JSON_BODY = 32 * 1024
WEB_ROOT = Path(__file__).with_name("web")
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/core.js": ("core.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")


def allowed_host(value: str) -> bool:
    try:
        split = urlsplit(f"//{value}")
        if split.port is not None and not 0 < split.port <= 65535:
            return False
        return (
            split.hostname in {"127.0.0.1", "localhost", "::1"}
            and not split.username
            and not split.password
            and not split.path
            and not split.query
            and not split.fragment
        )
    except ValueError:
        return False


def allowed_origin(value: str | None, host: str) -> bool:
    if value is None:
        return False
    try:
        origin = urlsplit(value)
        host_parts = urlsplit(f"//{host}")
        return (
            origin.scheme == "http"
            and origin.hostname in {"127.0.0.1", "localhost", "::1"}
            and origin.hostname == host_parts.hostname
            and origin.port == host_parts.port
            and not origin.username
            and not origin.password
            and origin.path in {"", "/"}
            and not origin.query
            and not origin.fragment
        )
    except ValueError:
        return False


def parse_range(value: str | None, size: int) -> tuple[int, int] | None:
    if value is None:
        return None
    match = RANGE_PATTERN.fullmatch(value.strip())
    if match is None or size <= 0:
        raise ValueError("invalid byte range")
    first, last = match.groups()
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("invalid byte range")
        return max(0, size - suffix), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if start >= size or end < start:
        raise ValueError("unsatisfiable byte range")
    return start, min(end, size - 1)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        manifest: Manifest,
        store: AnnotationStore,
    ):
        self.manifest = manifest
        self.store = store
        super().__init__(server_address, ReviewRequestHandler)


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server: ReviewServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        super().log_message(format, *args)

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
            "media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'",
        )
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")

    def _begin(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if self.close_connection:
            self.send_header("Connection", "close")
        self._security_headers()
        self.end_headers()

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self._begin(status, content_type, len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, value: Any) -> None:
        body = (_stable_json(value) + "\n").encode()
        self._send_bytes(status, "application/json; charset=utf-8", body)

    def _error(self, status: int, message: str) -> None:
        self.close_connection = True
        self._send_json(status, {"error": message})

    def _request_host_valid(self) -> bool:
        host = self.headers.get("Host", "")
        if allowed_host(host):
            return True
        self._error(HTTPStatus.MISDIRECTED_REQUEST, "Host must be loopback")
        return False

    def _origin_valid(self) -> bool:
        host = self.headers.get("Host", "")
        if allowed_origin(self.headers.get("Origin"), host):
            return True
        self._error(
            HTTPStatus.FORBIDDEN,
            "mutating requests require a same-origin loopback Origin",
        )
        return False

    def _json_body(self) -> Any:
        if self.headers.get_content_type() != "application/json":
            raise TypeError("Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Content-Length is invalid") from error
        if length < 1 or length > MAX_JSON_BODY:
            raise ValueError(f"JSON body must be between 1 and {MAX_JSON_BODY} bytes")
        try:
            return json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body is not valid JSON") from error

    def _serve_static(self, route: str) -> bool:
        entry = STATIC_ROUTES.get(route)
        if entry is None:
            return False
        name, content_type = entry
        body = (WEB_ROOT / name).read_bytes()
        self._send_bytes(HTTPStatus.OK, content_type, body)
        return True

    def _serve_media(self, media: MediaItem) -> None:
        size = media.path.stat().st_size
        try:
            byte_range = parse_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self._security_headers()
            self.end_headers()
            return
        if byte_range is None:
            start, end = 0, size - 1
            status = HTTPStatus.OK
        else:
            start, end = byte_range
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", media.mime_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self._security_headers()
        self.end_headers()
        if self.command == "HEAD" or length == 0:
            return
        with media.path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        if not self._request_host_valid():
            return
        route = unquote(urlsplit(self.path).path)
        if self._serve_static(route):
            return
        if route == "/api/manifest":
            self._send_json(HTTPStatus.OK, self.server.manifest.client_dict())
            return
        if route == "/api/annotations":
            self._send_json(HTTPStatus.OK, self.server.store.snapshot())
            return
        if route == "/api/export":
            body = self.server.store.export_jsonl().encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header(
                "Content-Disposition", 'attachment; filename="ground-truth.jsonl"'
            )
            self.send_header("Content-Length", str(len(body)))
            self._security_headers()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return
        prefix = "/media/"
        if route.startswith(prefix):
            token = route[len(prefix) :]
            if "/" not in token and token in self.server.manifest.media_by_token:
                self._serve_media(self.server.manifest.media_by_token[token])
                return
        self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        if not self._request_host_valid() or not self._origin_valid():
            return
        route = unquote(urlsplit(self.path).path)
        try:
            body = self._json_body()
        except TypeError as error:
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, str(error))
            return
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        if not isinstance(body, dict):
            self._error(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
            return
        try:
            if route == "/api/annotations":
                if set(body) != {"annotation", "expected_revision"}:
                    raise AnnotationError(
                        "save body must contain annotation and expected_revision"
                    )
                snapshot = self.server.store.upsert(
                    body["annotation"], body["expected_revision"]
                )
            elif route == "/api/archive":
                if set(body) != {"event_id", "expected_revision"}:
                    raise AnnotationError(
                        "archive body must contain event_id and expected_revision"
                    )
                snapshot = self.server.store.archive(
                    body["event_id"], body["expected_revision"]
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
        except RevisionConflict as error:
            self._error(HTTPStatus.CONFLICT, str(error))
            return
        except AnnotationError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._send_json(HTTPStatus.OK, snapshot)


def create_server(
    manifest_path: str | Path, annotation_path: str | Path, port: int = 8765
) -> ReviewServer:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    manifest = load_manifest(manifest_path)
    store = AnnotationStore(annotation_path, manifest)
    return ReviewServer(("127.0.0.1", port), manifest, store)
