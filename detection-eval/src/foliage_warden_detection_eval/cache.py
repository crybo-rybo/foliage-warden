"""Verified, atomic dataset downloads with explicit offline behavior."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .constants import (
    COCO_ANNOTATIONS_FILENAME,
    COCO_ANNOTATIONS_MD5,
    COCO_ANNOTATIONS_SHA256,
    COCO_ANNOTATIONS_SIZE,
    COCO_ANNOTATIONS_URL,
    COCO_INSTANCES_MEMBER,
    COCO_INSTANCES_SHA256,
)
from .errors import DetectionEvalError, OfflineCacheError

CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Content identity recorded for a verified artifact."""

    size: int
    md5: str
    sha256: str
    url: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "md5": self.md5,
            "sha256": self.sha256,
            "size": self.size,
            "url": self.url,
        }


def file_hashes(path: Path) -> tuple[int, str, str]:
    """Return size, MD5, and SHA-256 after one streaming read."""

    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                size += len(chunk)
                md5.update(chunk)
                sha256.update(chunk)
    except OSError as error:
        raise DetectionEvalError(f"cannot read cached artifact {path}: {error}") from error
    return size, md5.hexdigest(), sha256.hexdigest()


def sha256_file(path: Path) -> str:
    return file_hashes(path)[2]


def _verify(
    path: Path,
    *,
    expected_size: int,
    expected_md5: str,
    expected_sha256: str | None,
) -> ArtifactIdentity:
    size, md5, sha256 = file_hashes(path)
    failures: list[str] = []
    if size != expected_size:
        failures.append(f"size {size} != {expected_size}")
    if md5 != expected_md5:
        failures.append(f"MD5 {md5} != {expected_md5}")
    if expected_sha256 is not None and sha256 != expected_sha256:
        failures.append(f"SHA-256 {sha256} != {expected_sha256}")
    if failures:
        raise DetectionEvalError(f"artifact verification failed for {path}: {'; '.join(failures)}")
    return ArtifactIdentity(size=size, md5=md5, sha256=sha256, url="")


def _download(
    url: str,
    destination: Path,
    *,
    expected_size: int | None,
    expected_md5: str | None,
    expected_sha256: str | None,
) -> ArtifactIdentity:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "foliage-warden-detection-eval/1"},
            )
            try:
                response_context = urllib.request.urlopen(request, timeout=60)
            except OSError as error:
                raise DetectionEvalError(f"download failed for {url}: {error}") from error
            with response_context as response:
                header_size = response.headers.get("Content-Length")
                header_etag = response.headers.get("ETag", "").strip('"').lower()
                if expected_size is None:
                    if header_size is None or not header_size.isdecimal():
                        raise DetectionEvalError(
                            f"official server omitted Content-Length for {url}"
                        )
                    expected_size = int(header_size)
                elif header_size is not None and int(header_size) != expected_size:
                    raise DetectionEvalError(
                        f"official server reported unexpected size {header_size} for {url}; "
                        f"expected {expected_size}"
                    )
                if expected_md5 is None:
                    if len(header_etag) != 32 or any(
                        c not in "0123456789abcdef" for c in header_etag
                    ):
                        raise DetectionEvalError(
                            f"official server omitted a simple MD5 ETag for {url}; refusing "
                            "an unverified image download"
                        )
                    expected_md5 = header_etag
                elif header_etag and header_etag != expected_md5:
                    raise DetectionEvalError(
                        f"official server reported unexpected ETag {header_etag} for {url}; "
                        f"expected {expected_md5}"
                    )
                _copy_stream(response, output)

        assert temp_path is not None
        assert expected_size is not None
        assert expected_md5 is not None
        identity = _verify(
            temp_path,
            expected_size=expected_size,
            expected_md5=expected_md5,
            expected_sha256=expected_sha256,
        )
        os.replace(temp_path, destination)
        temp_path = None
        return ArtifactIdentity(
            size=identity.size,
            md5=identity.md5,
            sha256=identity.sha256,
            url=url,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    while chunk := source.read(CHUNK_SIZE):
        destination.write(chunk)


def ensure_pinned_artifact(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_md5: str,
    expected_sha256: str,
    offline: bool,
) -> ArtifactIdentity:
    """Use verified cached bytes or atomically download the exact pinned object."""

    if destination.is_file():
        try:
            verified = _verify(
                destination,
                expected_size=expected_size,
                expected_md5=expected_md5,
                expected_sha256=expected_sha256,
            )
            return ArtifactIdentity(
                size=verified.size,
                md5=verified.md5,
                sha256=verified.sha256,
                url=url,
            )
        except DetectionEvalError:
            if offline:
                raise OfflineCacheError(
                    f"offline cache entry is corrupt: {destination}; reconnect to replace it"
                ) from None
    elif offline:
        raise OfflineCacheError(f"offline cache miss: {destination}")
    return _download(
        url,
        destination,
        expected_size=expected_size,
        expected_md5=expected_md5,
        expected_sha256=expected_sha256,
    )


def ensure_remote_image(url: str, destination: Path, *, offline: bool) -> ArtifactIdentity:
    """Verify an image against its TLS-delivered S3 metadata and a local sidecar."""

    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    if destination.is_file() and metadata_path.is_file():
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if raw.get("url") != url:
                raise DetectionEvalError("cache URL differs from requested URL")
            expected_size = int(raw["size"])
            expected_md5 = str(raw["md5"])
            expected_sha256 = str(raw["sha256"])
            verified = _verify(
                destination,
                expected_size=expected_size,
                expected_md5=expected_md5,
                expected_sha256=expected_sha256,
            )
            return ArtifactIdentity(
                size=verified.size,
                md5=verified.md5,
                sha256=verified.sha256,
                url=url,
            )
        except (DetectionEvalError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            if offline:
                raise OfflineCacheError(
                    f"offline image cache entry is corrupt: {destination}; reconnect to replace it"
                ) from None
    elif offline:
        raise OfflineCacheError(
            f"offline image cache miss: {destination} (data and metadata sidecar are required)"
        )

    identity = _download(
        url,
        destination,
        expected_size=None,
        expected_md5=None,
        expected_sha256=None,
    )
    metadata_path.write_text(
        json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return identity


def ensure_coco_annotations(cache_dir: Path, *, offline: bool) -> Path:
    """Return the exact COCO 2017 val instances JSON from a verified archive."""

    archive = cache_dir / "downloads" / COCO_ANNOTATIONS_FILENAME
    ensure_pinned_artifact(
        COCO_ANNOTATIONS_URL,
        archive,
        expected_size=COCO_ANNOTATIONS_SIZE,
        expected_md5=COCO_ANNOTATIONS_MD5,
        expected_sha256=COCO_ANNOTATIONS_SHA256,
        offline=offline,
    )
    destination = cache_dir / COCO_INSTANCES_MEMBER
    if destination.is_file() and sha256_file(destination) == COCO_INSTANCES_SHA256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with zipfile.ZipFile(archive) as source_zip:
            try:
                member = source_zip.getinfo(COCO_INSTANCES_MEMBER)
            except KeyError as error:
                raise DetectionEvalError(
                    f"verified COCO archive lacks {COCO_INSTANCES_MEMBER}"
                ) from error
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as output:
                temp_path = Path(output.name)
                with source_zip.open(member) as source:
                    _copy_stream(source, output)
        assert temp_path is not None
        actual = sha256_file(temp_path)
        if actual != COCO_INSTANCES_SHA256:
            raise DetectionEvalError(
                f"extracted annotation SHA-256 {actual} != {COCO_INSTANCES_SHA256}"
            )
        os.replace(temp_path, destination)
        temp_path = None
        return destination
    except (OSError, zipfile.BadZipFile) as error:
        raise DetectionEvalError(f"cannot extract {archive}: {error}") from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
