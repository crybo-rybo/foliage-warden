from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from foliage_warden_detection_eval import cache
from foliage_warden_detection_eval.errors import OfflineCacheError


class _Response(io.BytesIO):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.headers = {
            "Content-Length": str(len(value)),
            "ETag": f'"{hashlib.md5(value, usedforsecurity=False).hexdigest()}"',
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_remote_image_cache_is_verified_and_reusable_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = b"not-a-real-jpeg-but-content-addressed"
    monkeypatch.setattr(cache.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(value))
    destination = tmp_path / "image.jpg"

    downloaded = cache.ensure_remote_image(
        "https://example.test/image.jpg", destination, offline=False
    )
    offline = cache.ensure_remote_image("https://example.test/image.jpg", destination, offline=True)

    assert downloaded == offline
    assert destination.read_bytes() == value
    metadata = json.loads((tmp_path / "image.jpg.metadata.json").read_text())
    assert metadata == downloaded.to_dict()


def test_corrupt_remote_image_cache_fails_closed_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = b"expected"
    monkeypatch.setattr(cache.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(value))
    destination = tmp_path / "image.jpg"
    cache.ensure_remote_image("https://example.test/image.jpg", destination, offline=False)
    destination.write_bytes(b"changed")

    with pytest.raises(OfflineCacheError, match="corrupt"):
        cache.ensure_remote_image("https://example.test/image.jpg", destination, offline=True)


def test_offline_cache_miss_has_actionable_failure(tmp_path: Path) -> None:
    with pytest.raises(OfflineCacheError, match="cache miss"):
        cache.ensure_remote_image(
            "https://example.test/image.jpg", tmp_path / "image.jpg", offline=True
        )


def test_pinned_artifact_cache_checks_all_identities(tmp_path: Path) -> None:
    value = b"pinned"
    destination = tmp_path / "artifact.zip"
    destination.write_bytes(value)

    identity = cache.ensure_pinned_artifact(
        "https://example.test/artifact.zip",
        destination,
        expected_size=len(value),
        expected_md5=hashlib.md5(value, usedforsecurity=False).hexdigest(),
        expected_sha256=hashlib.sha256(value).hexdigest(),
        offline=True,
    )

    assert identity.sha256 == hashlib.sha256(value).hexdigest()
