from __future__ import annotations

import http.client
import json
import threading
from http import HTTPStatus

from foliage_warden_review.server import (
    allowed_host,
    allowed_origin,
    create_server,
    parse_range,
)

from tests.support import ManifestTestCase


class ServerSecurityTests(ManifestTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.server = create_server(
            self.manifest_path, self.root / "annotations.json", 0
        )
        self.addCleanup(self.server.server_close)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._shutdown)
        self.host, self.port = self.server.server_address

    def _shutdown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)

    def request(
        self,
        method: str,
        path: str,
        *,
        body: str | None = None,
        headers: dict[str, str] | None = None,
        host: str | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=2)
        connection.putrequest(method, path, skip_host=True)
        connection.putheader("Host", host or f"{self.host}:{self.port}")
        for name, value in (headers or {}).items():
            connection.putheader(name, value)
        if body is not None:
            connection.putheader("Content-Length", str(len(body.encode())))
        connection.endheaders(body.encode() if body is not None else None)
        response = connection.getresponse()
        payload = response.read()
        result_headers = {name.lower(): value for name, value in response.getheaders()}
        connection.close()
        return response.status, result_headers, payload

    def test_server_is_loopback_and_static_traversal_is_not_served(self) -> None:
        self.assertEqual(self.host, "127.0.0.1")
        status, headers, _ = self.request("GET", "/")
        self.assertEqual(status, HTTPStatus.OK)
        self.assertIn("default-src 'self'", headers["content-security-policy"])
        for path in ("/../README.md", "/%2e%2e/README.md", "/media/../manifest.json"):
            with self.subTest(path=path):
                status, _, _ = self.request("GET", path)
                self.assertEqual(status, HTTPStatus.NOT_FOUND)

    def test_rejects_non_loopback_host_and_cross_origin_write(self) -> None:
        status, _, _ = self.request("GET", "/", host="attacker.example")
        self.assertEqual(status, HTTPStatus.MISDIRECTED_REQUEST)
        body = json.dumps({"annotation": self.annotation(), "expected_revision": 0})
        status, _, _ = self.request(
            "POST",
            "/api/annotations",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://attacker.example",
            },
        )
        self.assertEqual(status, HTTPStatus.FORBIDDEN)
        self.assertEqual(self.server.store.snapshot()["annotations"], [])

    def test_mutation_requires_json_and_same_loopback_origin(self) -> None:
        body = json.dumps({"annotation": self.annotation(), "expected_revision": 0})
        status, _, _ = self.request(
            "POST",
            "/api/annotations",
            body=body,
            headers={
                "Content-Type": "text/plain",
                "Origin": f"http://{self.host}:{self.port}",
            },
        )
        self.assertEqual(status, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        status, _, payload = self.request(
            "POST",
            "/api/annotations",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Origin": f"http://{self.host}:{self.port}",
            },
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(json.loads(payload)["revision"], 1)

    def test_media_is_addressed_by_token_and_supports_bounded_ranges(self) -> None:
        token = next(iter(self.server.manifest.media_by_token))
        status, headers, payload = self.request(
            "GET", f"/media/{token}", headers={"Range": "bytes=0-4"}
        )
        self.assertEqual(status, HTTPStatus.PARTIAL_CONTENT)
        self.assertEqual(headers["content-range"], "bytes 0-4/19")
        self.assertEqual(payload, b"local")
        status, _, _ = self.request("GET", "/media/not-a-token")
        self.assertEqual(status, HTTPStatus.NOT_FOUND)

    def test_host_origin_and_range_parsers_are_fail_closed(self) -> None:
        self.assertTrue(allowed_host("127.0.0.1:8765"))
        self.assertTrue(allowed_origin("http://localhost:8765", "localhost:8765"))
        self.assertFalse(allowed_host("example.com"))
        self.assertFalse(allowed_host("attacker@127.0.0.1:8765"))
        self.assertFalse(allowed_host("127.0.0.1:8765/path"))
        self.assertFalse(allowed_origin("https://localhost:8765", "localhost:8765"))
        self.assertEqual(parse_range("bytes=-3", 10), (7, 9))
        with self.assertRaises(ValueError):
            parse_range("bytes=10-20", 10)


if __name__ == "__main__":
    import unittest

    unittest.main()
